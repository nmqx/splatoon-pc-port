// gx2_vk_framegen.inl - optical-flow frame generation ("frame gen"), included
// by gx2_vk.cpp inside namespace gx2vk (it uses that file's statics).
//
// The game draws 60 (or 30) frames a second; the display refreshes faster
// (164 Hz here). With settings.ini frame_gen=1 (ESC menu; RBREW_FRAMEGEN=0/1
// wins) one picture is shown per display refresh, made between the game's
// two last frames at the point in time where that refresh falls. The game
// itself is not sped up: its cadence is the emulated vblank clock
// (GX2WaitForVsync / pace_flip sleep on the host clock), never the present.
//
// Per game frame, render thread (gx2vk::present -> present_hook):
//   1. the scene image is copied 1:1 into col[k] (B8G8R8A8, a ring of
//      kSlots) and, above flow_lines() lines, into a smaller copy cols[k] the
//      flow is measured on; the ImGui overlay (menu, FPS) is drawn into
//      ovl[k] (swapchain format and extent, cleared to transparent: ImGui's
//      blend leaves premultiplied colour and coverage). One submission on
//      s_queue, timeline value s_timeline = serial.
//   2. optical flow queue (its own family): forward and backward flow between
//      col[k] and col[k-1] on a 4x4 grid (S10.5), copied into buf[k]; waits
//      for s_timeline >= serial, signals of_tl.
//   3. the slot goes to the presenter thread.
// The presenter thread (its own queue, graphics family index 1) owns the
// swapchain (FIFO) while frame gen is on. A little before each vertical blank
// of DWM's timing it places a playhead about one game frame behind the
// newest frame, and a compute shader makes the picture there: the pair of
// samples of the two frames that the forward flow, the backward flow or no
// motion join best, mixed by alpha, stretched to the swapchain extent like
// the normal LINEAR blit, with the overlay composited (ovl + scene * (1 - a)).
// The picture is blitted to the swapchain image and presented.
//
// The render thread touches the swapchain only with the presenter paused
// (pause(): it gives back every frame and does no GPU work until the next
// frame is pushed); swapchain or size changes then also wait for the device.
// Anything unsupported (no VK_NV_optical_flow, no second graphics queue, no
// optical flow family, a session that cannot be made) is logged and the
// normal presentation stays.
#if defined(_WIN32)
namespace {
uint64_t submit_with(const VkSemaphore* wait, VkPipelineStageFlags2 wait_stage, const VkSemaphore* signal);

namespace fg {

constexpr int kSlots = 6;      // frames in the ring
constexpr int kHold = 3;       // frames the presenter keeps (the playhead lags 1..2 frames): the render
                               // thread's next slot was given back two frames ago
constexpr int kQueue = 16;     // items given to the presenter, not yet taken
constexpr int kCmds = 4;       // presenter command buffers in flight
constexpr int kAcquire = 8;    // acquire semaphores
constexpr int kMaxImages = 16; // swapchain images (present semaphores)

// Driver entry points. The volk pointers of the vkCmd*/vkQueue* functions are
// the split render thread's wrappers (record::install): they queue commands
// for the recording thread of s_current. Command buffers of the optical flow
// queue and of the presenter thread are recorded with these instead.
struct Fn {
    PFN_vkCreateOpticalFlowSessionNV CreateOpticalFlowSessionNV;
    PFN_vkDestroyOpticalFlowSessionNV DestroyOpticalFlowSessionNV;
    PFN_vkBindOpticalFlowSessionImageNV BindOpticalFlowSessionImageNV;
    PFN_vkCmdOpticalFlowExecuteNV CmdOpticalFlowExecuteNV;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkResetCommandBuffer ResetCommandBuffer;
    PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdPushDescriptorSetKHR CmdPushDescriptorSetKHR;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkQueueSubmit2 QueueSubmit2;
    PFN_vkQueuePresentKHR QueuePresentKHR;
};
Fn fn{};

struct Img {
    VkImage image = VK_NULL_HANDLE;
    Allocation memory;
    VkImageView view = VK_NULL_HANDLE;
};
struct Item { // a game frame given to the presenter
    int slot;
    bool flow;          // its flow to the frame before was measured
    uint64_t wait;      // s_timeline value: the copy and the overlay are done
    uint64_t flow_wait; // of_tl value: buf[slot] holds the flow
    int64_t t;          // QPC when it was given
};
struct Held {
    int slot;
    bool flow;
    uint64_t wait, flow_wait;
    int64_t t, n; // n: frame number (consecutive while held)
};
struct Pick { // what the presenter shows at a refresh
    int prev = -1, cur = -1; // slots; prev < 0: cur as it is
    int mode = 1;            // shader mode: 0 between (flow + still), 1 cur as it is, 2 flow only, 3 still only
    float alpha = 0.0f;
    uint64_t wait = 0, flow_wait = 0;
};

struct State {
    // device (init)
    bool dev = false; // extension, feature and queues are part of the device
    uint32_t of_family = 0;
    VkPhysicalDeviceOpticalFlowFeaturesNV feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV};
    void** link = nullptr; // the pNext the feature is chained to
    VkDeviceQueueCreateInfo qi[2]{};
    float gprio[2] = {1.0f, 1.0f}, oprio = 1.0f;
    VkQueue of_queue = VK_NULL_HANDLE, p_queue = VK_NULL_HANDLE;
    uint32_t of_min_w = 0, of_min_h = 0, of_max_w = 0, of_max_h = 0; // optical flow image size limits
    // objects kept from the first use to shutdown
    bool objects = false, failed = false, on = false;
    VkSampler smp = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkSemaphore of_tl = VK_NULL_HANDLE, ptl = VK_NULL_HANDLE; // timelines: flow done, presenter submissions
    uint64_t of_val = 0;
    VkCommandPool of_pool = VK_NULL_HANDLE, p_pool = VK_NULL_HANDLE;
    // presenter: per picture a compute submission (2 i: reads the frames,
    // signals ptl, never waits for the display) and a blit submission (2 i + 1:
    // waits for the acquired image)
    VkCommandBuffer of_cmd[kSlots]{}, p_cmd[2 * kCmds]{};
    VkFence p_fence[2 * kCmds]{};
    bool p_pending[2 * kCmds]{};
    uint64_t p_read[kSlots]{}; // presenter thread: ptl value of the last compute that read the slot
    VkSemaphore p_acq[kAcquire]{}, p_rend[kMaxImages]{};
    int p_i = 0, p_ai = 0, out_i = 0;
    uint64_t pval = 0; // presenter thread: last ptl value submitted
    // frames (source size)
    uint32_t w = 0, h = 0, fw = 0, fh = 0, gw = 0, gh = 0;
    float fs = 1.0f;
    bool reduced = false, flow_init = false;
    Img col[kSlots], cols[kSlots], flow[2];
    VkBuffer buf[kSlots]{};
    Allocation buf_mem[kSlots];
    VkOpticalFlowSessionNV session[kSlots]{};
    uint64_t of_used[kSlots]{}; // of_tl value of the last flow run of session k (reads col[k], col[k - 1])
    // the swapchain the presenter presents to (a copy: the presenter never reads
    // s_swapchain, whose vector may be gone while it runs at process exit)
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    VkImage sc_images[kMaxImages]{};
    uint32_t sc_count = 0;
    // swapchain-sized
    VkExtent2D ext{};
    VkFormat ovl_format = VK_FORMAT_UNDEFINED;
    Img ovl[kSlots], out[2];
    bool out_init[2]{};
    bool has_ovl[kSlots]{};
    // render thread
    int k = 0;
    bool have_prev = false;
    int64_t t_last = 0;
    double period = 1000.0 / 60.0; // game frame interval, ms (smoothed)
    uint64_t serials[3]{};
    unsigned serial_i = 0;
    uint64_t frames = 0;
    // presenter thread and what it shares (under lock)
    HANDLE thread = nullptr;
    SRWLOCK lock = SRWLOCK_INIT;
    CONDITION_VARIABLE cv = CONDITION_VARIABLE_INIT, idle_cv = CONDITION_VARIABLE_INIT;
    Item q[kQueue]{};
    int qh = 0, qn = 0, held = 0;
    bool busy = false, stop = false, paused = false;
    int slot_items[kSlots]{};   // queued or held by the presenter
    uint64_t p_used[kSlots]{}; // ptl value after which the presenter no longer reads the slot
    double shared_period = 1000.0 / 60.0;
    std::atomic<bool> swap_bad{false};
    std::atomic<int64_t> shown{0}, made{0}, late{0}, skipped{0};
    int debug = 0; // RBREW_FRAMEGEN_DEBUG: 1 newest frame as it is, 2 flow only at alpha 0.5, 3 still blend only
    // RBREW_FRAMEGEN_LOG=<file>: per-second statistics (written by the render thread)
    struct Stats {
        std::atomic<int64_t> frames{0}, shown{0}, made{0}, as_is{0}, late{0}, skipped{0}, dropped{0}, resets{0};
        std::atomic<int64_t> hook_us{0}, hook_max{0}, wait_us{0}, wait_max{0}, slot_timeouts{0};
        std::atomic<int64_t> acq_us{0}, acq_max{0}, pres_us{0}, pres_max{0}, swap_bad{0};
        std::atomic<int64_t> fb_capture{0}, fb_source{0}, fb_minimized{0}, recreate{0}, rebuild{0}, pauses{0};
        std::atomic<int64_t> alpha[8]{};
        std::atomic<int> lag_milli{1500};
    } st;
};
State G;

int64_t qpc() {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return v.QuadPart;
}
int64_t qpf() {
    static const int64_t f = [] {
        LARGE_INTEGER v;
        QueryPerformanceFrequency(&v);
        return v.QuadPart;
    }();
    return f;
}
void sleep_until_qpc(int64_t t) {
    const int64_t now = qpc();
    if (t <= now) return;
    rbrew_host_sleep_until(std::chrono::steady_clock::now() +
                           std::chrono::nanoseconds((t - now) * 1000000000ll / qpf()));
}
uint64_t counter(VkSemaphore s) {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(s_device, s, &v);
    return v;
}
bool wait_value(VkSemaphore s, uint64_t v, uint64_t timeout_ns) {
    if (!v) return true;
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1;
    info.pSemaphores = &s;
    info.pValues = &v;
    return vkWaitSemaphores(s_device, &info, timeout_ns) == VK_SUCCESS;
}
void image_barrier(PFN_vkCmdPipelineBarrier2 f, VkCommandBuffer cb, VkImage image, VkImageLayout from,
                   VkImageLayout to, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                   VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = ss;
    b.srcAccessMask = sa;
    b.dstStageMask = ds;
    b.dstAccessMask = da;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    f(cb, &d);
}

// ---------------------------------------------------------------------------
// Device creation (init)
// ---------------------------------------------------------------------------
// Before vkCreateDevice: the extension and its feature (chained at *tail).
void device_prepare(const std::vector<VkExtensionProperties>& extensions, std::vector<const char*>& enable,
                    void**& tail, bool push_descriptors) {
    G.dev = false;
    if (!s_surface || !push_descriptors) return;
    if (const char* v = std::getenv("RBREW_FRAMEGEN_DEVICE"); v && v[0] == '0') return;
    if (!has_extension(extensions, VK_NV_OPTICAL_FLOW_EXTENSION_NAME)) {
        RBREW_LOG_INFO("vk", "frame gen: unavailable (no VK_NV_optical_flow: NVIDIA RTX 30 and later)\n");
        return;
    }
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(s_physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(s_physical, &count, families.data());
    if (s_queue_family >= count || families[s_queue_family].queueCount < 2) {
        RBREW_LOG_INFO("vk", "frame gen: unavailable (the graphics family has a single queue)\n");
        return;
    }
    int family = -1;
    for (uint32_t i = 0; i < count && family < 0; ++i)
        if (i != s_queue_family && (families[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) &&
            (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT))
            family = int(i);
    if (family < 0) {
        RBREW_LOG_INFO("vk", "frame gen: unavailable (no queue family with optical flow and transfer)\n");
        return;
    }
    VkPhysicalDeviceOpticalFlowFeaturesNV query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &query;
    vkGetPhysicalDeviceFeatures2(s_physical, &f2);
    if (!query.opticalFlow) {
        RBREW_LOG_INFO("vk", "frame gen: unavailable (the opticalFlow feature is not offered)\n");
        return;
    }
    G.of_family = uint32_t(family);
    enable.push_back(VK_NV_OPTICAL_FLOW_EXTENSION_NAME);
    G.feat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV};
    G.feat.opticalFlow = VK_TRUE;
    *tail = &G.feat;
    G.link = tail;
    tail = &G.feat.pNext;
    G.dev = true;
}
// The presenter's queue (graphics family, index 1) and the optical flow queue.
void device_queues(VkDeviceCreateInfo& info, const VkDeviceQueueCreateInfo& graphics) {
    if (!G.dev) return;
    G.qi[0] = graphics;
    G.qi[0].queueCount = 2;
    G.qi[0].pQueuePriorities = G.gprio;
    G.qi[1] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    G.qi[1].queueFamilyIndex = G.of_family;
    G.qi[1].queueCount = 1;
    G.qi[1].pQueuePriorities = &G.oprio;
    info.queueCreateInfoCount = 2;
    info.pQueueCreateInfos = G.qi;
}
// vkCreateDevice failed with the additions: the device without them.
bool device_revert(std::vector<const char*>& enable, VkDeviceCreateInfo& info,
                   const VkDeviceQueueCreateInfo* graphics) {
    if (!G.dev) return false;
    enable.erase(std::remove_if(enable.begin(), enable.end(),
                                [](const char* e) { return !std::strcmp(e, VK_NV_OPTICAL_FLOW_EXTENSION_NAME); }),
                 enable.end());
    if (G.link) *G.link = nullptr;
    G.link = nullptr;
    info.enabledExtensionCount = uint32_t(enable.size());
    info.ppEnabledExtensionNames = enable.data();
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = graphics;
    G.dev = false;
    RBREW_LOG_WARN("vk", "frame gen: the device with optical flow could not be created; retrying without it\n");
    return true;
}
void device_created() {
    if (!G.dev) return;
    G.link = nullptr;
    vkGetDeviceQueue(s_device, s_queue_family, 1, &G.p_queue);
    vkGetDeviceQueue(s_device, G.of_family, 0, &G.of_queue);
    bool ok = G.p_queue && G.of_queue;
#define FG_LOAD(Name)                                                                                                  \
    fn.Name = reinterpret_cast<PFN_vk##Name>(vkGetDeviceProcAddr(s_device, "vk" #Name));                              \
    ok = ok && fn.Name != nullptr;
    FG_LOAD(CreateOpticalFlowSessionNV)
    FG_LOAD(DestroyOpticalFlowSessionNV)
    FG_LOAD(BindOpticalFlowSessionImageNV)
    FG_LOAD(CmdOpticalFlowExecuteNV)
    FG_LOAD(AllocateCommandBuffers)
    FG_LOAD(BeginCommandBuffer)
    FG_LOAD(EndCommandBuffer)
    FG_LOAD(ResetCommandBuffer)
    FG_LOAD(CmdPipelineBarrier2)
    FG_LOAD(CmdCopyImageToBuffer)
    FG_LOAD(CmdBlitImage)
    FG_LOAD(CmdBindPipeline)
    FG_LOAD(CmdPushDescriptorSetKHR)
    FG_LOAD(CmdPushConstants)
    FG_LOAD(CmdDispatch)
    FG_LOAD(QueueSubmit2)
    FG_LOAD(QueuePresentKHR)
#undef FG_LOAD
    VkPhysicalDeviceOpticalFlowPropertiesNV of{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &of;
    vkGetPhysicalDeviceProperties2(s_physical, &p2);
    G.of_min_w = std::max(of.minWidth, 64u);
    G.of_min_h = std::max(of.minHeight, 64u);
    G.of_max_w = of.maxWidth;
    G.of_max_h = of.maxHeight;
    if (!(of.supportedOutputGridSizes & VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV) || !of.bidirectionalFlowSupported) {
        RBREW_LOG_WARN("vk", "frame gen: unavailable (no 4x4 grid or no flow in both directions)\n");
        ok = false;
    }
    if (!ok) {
        RBREW_LOG_WARN("vk", "frame gen: unavailable (device queues or functions missing)\n");
        G.dev = false;
        return;
    }
    RBREW_LOG_INFO("vk", "frame gen: available (optical flow queue family %u, presenter queue %u/1)\n", G.of_family,
                   s_queue_family);
}

// ---------------------------------------------------------------------------
// The compute stage
// ---------------------------------------------------------------------------
// The picture at alpha t between prev (0) and cur (1), at the output pixel q
// of the swapchain-sized picture (the scene is stretched as the LINEAR blit
// does). Forward flow (cur -> prev: cur(x) ~ prev(x + fw(x))) carries what is
// at p from p + t fw in prev to p - (1 - t) fw in cur; the backward flow
// (prev(x) ~ cur(x + bw(x))) from p - t bw to p + (1 - t) bw. Of those two
// pairs and the still pair, the one whose two samples agree best is mixed.
const char* const kShader = R"(#version 450
layout(local_size_x = 16, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D prevTex;
layout(set = 0, binding = 1) uniform sampler2D curTex;
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D outImg;
layout(set = 0, binding = 3, std430) readonly buffer Flow { int v[]; } flow;
layout(set = 0, binding = 4) uniform sampler2D ovlTex;
layout(push_constant) uniform Pc {
    ivec2 outSize;
    ivec2 srcSize;
    ivec2 grid;
    int mode;   // 0: between prev and cur, 1: cur as it is, 2: flow pairs only, 3: still pair only
    int hasOvl;
    float fs;   // source pixels per flow pixel
    float alpha;
} pc;
vec2 fetchFlow(int base, ivec2 c) {
    c = clamp(c, ivec2(0), pc.grid - 1);
    int w = flow.v[base + c.y * pc.grid.x + c.x];
    return vec2(float((w << 16) >> 16), float(w >> 16)) * (1.0 / 32.0);
}
vec2 flowAt(int base, vec2 p) {
    vec2 g = p / pc.fs * 0.25 - 0.5;
    ivec2 c = ivec2(floor(g));
    vec2 f = g - vec2(c);
    return pc.fs * mix(mix(fetchFlow(base, c), fetchFlow(base, c + ivec2(1, 0)), f.x),
                       mix(fetchFlow(base, c + ivec2(0, 1)), fetchFlow(base, c + ivec2(1, 1)), f.x), f.y);
}
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= pc.outSize.x || q.y >= pc.outSize.y) return;
    vec2 uv = (vec2(q) + 0.5) / vec2(pc.outSize);
    vec3 r;
    if (pc.mode == 1) {
        r = texture(curTex, uv).rgb;
    } else {
        vec2 inv = 1.0 / vec2(pc.srcSize);
        vec2 p = uv * vec2(pc.srcSize);
        int n = pc.grid.x * pc.grid.y;
        float t = pc.alpha, u = 1.0 - t;
        // The vectors are read where the motion starts, not at p: the forward
        // field lives on cur's pixels (what is at p at time t is at x in cur,
        // x + u fw(x) = p), the backward one on prev's (y + t bw(y) = p). Two
        // fixed-point steps; reading them at p tore edges of moving objects.
        vec2 fw = flowAt(0, p);
        fw = flowAt(0, p - u * fw);
        fw = flowAt(0, p - u * fw);
        vec2 bw = flowAt(n, p);
        bw = flowAt(n, p - t * bw);
        bw = flowAt(n, p - t * bw);
        vec3 a0 = texture(prevTex, (p + t * fw) * inv).rgb, a1 = texture(curTex, (p - u * fw) * inv).rgb;
        vec3 b0 = texture(prevTex, (p - t * bw) * inv).rgb, b1 = texture(curTex, (p + u * bw) * inv).rgb;
        vec3 c0 = texture(prevTex, uv).rgb, c1 = texture(curTex, uv).rgb;
        float ea = dot(abs(a0 - a1), vec3(1.0)), eb = dot(abs(b0 - b1), vec3(1.0));
        float ec = max(dot(abs(c0 - c1), vec3(1.0)) - 0.02, 0.0);
        // A pair whose samples disagree has one side hidden (occlusion): its
        // nearer frame's sample instead of a double image.
        vec3 ra = mix(mix(a0, a1, t), t < 0.5 ? a0 : a1, smoothstep(0.15, 0.6, ea));
        vec3 rb = mix(mix(b0, b1, t), t < 0.5 ? b0 : b1, smoothstep(0.15, 0.6, eb));
        vec3 rc = mix(c0, c1, t);
        if (pc.mode == 3) {
            r = rc;
        } else {
            // Soft weights: picking one pair per pixel by its error flipped
            // neighbouring pixels between pairs (speckle, shimmering edges).
            float wa = exp(-24.0 * ea), wb = exp(-24.0 * eb), wc = pc.mode == 2 ? 0.0 : 1.5 * exp(-24.0 * ec);
            float sum = wa + wb + wc;
            r = sum > 1e-6 ? (wa * ra + wb * rb + wc * rc) / sum : (ea <= eb ? ra : rb);
        }
    }
    if (pc.hasOvl != 0) {
        vec4 o = texelFetch(ovlTex, q, 0);
        r = o.rgb + r * (1.0 - o.a);
    }
    imageStore(outImg, q, vec4(r, 1.0));
}
)";
struct PushConstants {
    int32_t out_w, out_h, src_w, src_h, grid_w, grid_h, mode, has_ovl;
    float fs, alpha;
};
static_assert(sizeof(PushConstants) == 40);

void destroy_objects();
bool make_objects() {
    G.objects = true; // destroy_objects() frees what was made on failure
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(s_device, &si, nullptr, &G.smp) != VK_SUCCESS) return false;
    const VkDescriptorType types[5] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (uint32_t i = 0; i < 5; ++i) bindings[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = 5;
    li.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(s_device, &li, nullptr, &G.dsl) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &G.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(s_device, &pli, nullptr, &G.pl) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = shader_module(VK_SHADER_STAGE_COMPUTE_BIT, kShader);
    ci.stage.pName = "main";
    ci.layout = G.pl;
    if (!ci.stage.module ||
        vkCreateComputePipelines(s_device, pipeline_cache(), 1, &ci, nullptr, &G.pipe) != VK_SUCCESS) {
        G.pipe = VK_NULL_HANDLE;
        return false;
    }
    VkSemaphoreTypeCreateInfo tt{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tt.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ti.pNext = &tt;
    if (vkCreateSemaphore(s_device, &ti, nullptr, &G.of_tl) != VK_SUCCESS ||
        vkCreateSemaphore(s_device, &ti, nullptr, &G.ptl) != VK_SUCCESS)
        return false;
    VkSemaphoreCreateInfo bs{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : G.p_acq)
        if (vkCreateSemaphore(s_device, &bs, nullptr, &s) != VK_SUCCESS) return false;
    for (auto& s : G.p_rend)
        if (vkCreateSemaphore(s_device, &bs, nullptr, &s) != VK_SUCCESS) return false;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto& f : G.p_fence)
        if (vkCreateFence(s_device, &fi, nullptr, &f) != VK_SUCCESS) return false;
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = G.of_family;
    if (vkCreateCommandPool(s_device, &pi, nullptr, &G.of_pool) != VK_SUCCESS) return false;
    pi.queueFamilyIndex = s_queue_family;
    if (vkCreateCommandPool(s_device, &pi, nullptr, &G.p_pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool = G.of_pool;
    ai.commandBufferCount = kSlots;
    if (fn.AllocateCommandBuffers(s_device, &ai, G.of_cmd) != VK_SUCCESS) return false;
    ai.commandPool = G.p_pool;
    ai.commandBufferCount = 2 * kCmds;
    if (fn.AllocateCommandBuffers(s_device, &ai, G.p_cmd) != VK_SUCCESS) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Images (render thread; the device is idle when they are freed)
// ---------------------------------------------------------------------------
bool make_img(Img& r, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, bool concurrent,
              VkOpticalFlowUsageFlagsNV of_usage) {
    const uint32_t families[2] = {s_queue_family, G.of_family};
    VkOpticalFlowImageFormatInfoNV of{VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV};
    of.usage = of_usage;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.pNext = of_usage ? &of : nullptr;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {w, h, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    info.queueFamilyIndexCount = concurrent ? 2 : 0;
    info.pQueueFamilyIndices = concurrent ? families : nullptr;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(s_device, &info, nullptr, &r.image) != VK_SUCCESS) {
        r.image = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(s_device, r.image, &req);
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.image = r.image;
    r.memory = allocate(req, MemoryUse::DeviceLocal, false, &dedicated);
    if (!r.memory || vkBindImageMemory(s_device, r.image, r.memory.memory, r.memory.offset) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = r.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(s_device, &vi, nullptr, &r.view) == VK_SUCCESS;
}
void free_img(Img& r) {
    if (r.view) vkDestroyImageView(s_device, r.view, nullptr);
    if (r.image) vkDestroyImage(s_device, r.image, nullptr);
    free_now(r.memory);
    r = Img{};
}

void free_slots() {
    for (int i = 0; i < kSlots; ++i) {
        if (G.session[i]) fn.DestroyOpticalFlowSessionNV(s_device, G.session[i], nullptr);
        G.session[i] = VK_NULL_HANDLE;
        free_img(G.col[i]);
        free_img(G.cols[i]);
        if (G.buf[i]) vkDestroyBuffer(s_device, G.buf[i], nullptr);
        G.buf[i] = VK_NULL_HANDLE;
        free_now(G.buf_mem[i]);
    }
    for (auto& f : G.flow) free_img(f);
    G.w = G.h = 0;
    G.have_prev = false;
    G.flow_init = false;
}
// Frames taller than this are measured on a reduced copy (RBREW_FRAMEGEN_FLOW_H).
uint32_t flow_lines() {
    static const uint32_t lines = [] {
        const char* v = std::getenv("RBREW_FRAMEGEN_FLOW_H");
        return v && std::atoi(v) >= 180 ? uint32_t(std::atoi(v)) : 900u;
    }();
    return lines;
}
// The frames of w x h: ring images, flow images and buffers, the sessions.
bool make_slots(uint32_t w, uint32_t h) {
    // Above flow_lines() lines the flow is measured on a reduced copy: at full
    // size (1440 lines) the optical flow engine took ~5 ms a frame; 4x4 cells
    // of the smaller copy are as good for the picture, still made full size.
    const uint32_t lines = flow_lines();
    G.reduced = h > lines;
    G.fh = G.reduced ? lines : h;
    G.fw = G.reduced ? uint32_t(double(w) * lines / h + 0.5) : w;
    G.fs = float(w) / float(G.fw);
    G.gw = (G.fw + 3) / 4;
    G.gh = (G.fh + 3) / 4;
    G.w = w;
    G.h = h;
    const uint32_t families[2] = {s_queue_family, G.of_family};
    for (int i = 0; i < kSlots; ++i) {
        if (!make_img(G.col[i], w, h, VK_FORMAT_B8G8R8A8_UNORM,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      true, VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV))
            return false;
        if (G.reduced && !make_img(G.cols[i], G.fw, G.fh, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT, true,
                                   VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV))
            return false;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = VkDeviceSize(G.gw) * G.gh * 8; // forward, then backward
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2;
        bi.pQueueFamilyIndices = families;
        if (vkCreateBuffer(s_device, &bi, nullptr, &G.buf[i]) != VK_SUCCESS) {
            G.buf[i] = VK_NULL_HANDLE;
            return false;
        }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(s_device, G.buf[i], &req);
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.buffer = G.buf[i];
        G.buf_mem[i] = allocate(req, MemoryUse::DeviceLocal, true, &dedicated);
        if (!G.buf_mem[i] ||
            vkBindBufferMemory(s_device, G.buf[i], G.buf_mem[i].memory, G.buf_mem[i].offset) != VK_SUCCESS)
            return false;
    }
    for (auto& f : G.flow)
        if (!make_img(f, G.gw, G.gh, VK_FORMAT_R16G16_SFIXED5_NV, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false,
                      VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV))
            return false;
    for (int i = 0; i < kSlots; ++i) {
        VkOpticalFlowSessionCreateInfoNV ci{VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV};
        ci.width = G.fw;
        ci.height = G.fh;
        ci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
        ci.flowVectorFormat = VK_FORMAT_R16G16_SFIXED5_NV;
        ci.outputGridSize = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV;
        // RBREW_FRAMEGEN_OF=slow|medium|fast: the engine's quality / speed trade (medium by default)
        const char* level = std::getenv("RBREW_FRAMEGEN_OF");
        ci.performanceLevel = level && !std::strcmp(level, "slow")   ? VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV
                              : level && !std::strcmp(level, "fast") ? VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_NV
                                                                     : VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV;
        ci.flags = VK_OPTICAL_FLOW_SESSION_CREATE_BOTH_DIRECTIONS_BIT_NV;
        const VkResult r = fn.CreateOpticalFlowSessionNV(s_device, &ci, nullptr, &G.session[i]);
        if (r != VK_SUCCESS) {
            G.session[i] = VK_NULL_HANDLE;
            RBREW_LOG_WARN("vk", "frame gen: optical flow session %ux%u failed (%d)\n", G.fw, G.fh, int(r));
            return false;
        }
        const int p = (i + kSlots - 1) % kSlots; // reference: the frame before
        const Img* in = G.reduced ? G.cols : G.col;
        if (fn.BindOpticalFlowSessionImageNV(s_device, G.session[i], VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
                                             in[i].view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS ||
            fn.BindOpticalFlowSessionImageNV(s_device, G.session[i],
                                             VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV, in[p].view,
                                             VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS ||
            fn.BindOpticalFlowSessionImageNV(s_device, G.session[i],
                                             VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV, G.flow[0].view,
                                             VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS ||
            fn.BindOpticalFlowSessionImageNV(s_device, G.session[i],
                                             VK_OPTICAL_FLOW_SESSION_BINDING_POINT_BACKWARD_FLOW_VECTOR_NV,
                                             G.flow[1].view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS)
            return false;
    }
    // The ring images live in GENERAL (recorded before this frame's copy).
    VkCommandBuffer cb = cmd();
    for (int i = 0; i < kSlots; ++i) {
        image_barrier(vkCmdPipelineBarrier2, cb, G.col[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        if (G.reduced)
            image_barrier(vkCmdPipelineBarrier2, cb, G.cols[i].image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    }
    G.flow_init = false;
    G.have_prev = false;
    G.k = 0;
    RBREW_LOG_INFO("vk", "frame gen: frames of %ux%u, flow on %ux%u (%s), grid %ux%u\n", w, h, G.fw, G.fh,
                   G.reduced ? "reduced copy" : "the frames", G.gw, G.gh);
    return true;
}

void free_screen() {
    for (auto& o : G.ovl) free_img(o);
    for (auto& o : G.out) free_img(o);
    for (auto& b : G.has_ovl) b = false;
    G.out_init[0] = G.out_init[1] = false;
    G.ext = {};
    G.ovl_format = VK_FORMAT_UNDEFINED;
}
bool make_screen(VkExtent2D ext, VkFormat format) {
    G.ext = ext;
    G.ovl_format = format;
    for (auto& o : G.ovl)
        if (!make_img(o, ext.width, ext.height, format,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                          VK_IMAGE_USAGE_SAMPLED_BIT,
                      false, 0))
            return false;
    for (auto& o : G.out)
        if (!make_img(o, ext.width, ext.height, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false, 0))
            return false;
    for (auto& b : G.has_ovl) b = false;
    G.out_init[0] = G.out_init[1] = false;
    return true;
}

void destroy_objects() {
    if (!G.objects) return;
    if (G.pipe) vkDestroyPipeline(s_device, G.pipe, nullptr);
    if (G.pl) vkDestroyPipelineLayout(s_device, G.pl, nullptr);
    if (G.dsl) vkDestroyDescriptorSetLayout(s_device, G.dsl, nullptr);
    if (G.smp) vkDestroySampler(s_device, G.smp, nullptr);
    if (G.of_tl) vkDestroySemaphore(s_device, G.of_tl, nullptr);
    if (G.ptl) vkDestroySemaphore(s_device, G.ptl, nullptr);
    for (auto& s : G.p_acq)
        if (s) vkDestroySemaphore(s_device, s, nullptr), s = VK_NULL_HANDLE;
    for (auto& s : G.p_rend)
        if (s) vkDestroySemaphore(s_device, s, nullptr), s = VK_NULL_HANDLE;
    for (auto& f : G.p_fence)
        if (f) vkDestroyFence(s_device, f, nullptr), f = VK_NULL_HANDLE;
    // Destroying a pool frees its command buffers (the real function: no
    // drain of the render thread's recording ring is involved).
    if (G.of_pool) record::real.DestroyCommandPool ? record::real.DestroyCommandPool(s_device, G.of_pool, nullptr)
                                                   : vkDestroyCommandPool(s_device, G.of_pool, nullptr);
    if (G.p_pool) record::real.DestroyCommandPool ? record::real.DestroyCommandPool(s_device, G.p_pool, nullptr)
                                                  : vkDestroyCommandPool(s_device, G.p_pool, nullptr);
    G.pipe = VK_NULL_HANDLE;
    G.pl = VK_NULL_HANDLE;
    G.dsl = VK_NULL_HANDLE;
    G.smp = VK_NULL_HANDLE;
    G.of_tl = G.ptl = VK_NULL_HANDLE;
    G.of_pool = G.p_pool = VK_NULL_HANDLE;
    for (auto& c : G.of_cmd) c = VK_NULL_HANDLE;
    for (auto& c : G.p_cmd) c = VK_NULL_HANDLE;
    for (auto& p : G.p_pending) p = false;
    for (auto& v : G.of_used) v = 0;
    for (auto& v : G.p_used) v = 0;
    for (auto& v : G.p_read) v = 0;
    G.of_val = G.pval = 0;
    G.objects = false;
}

// ---------------------------------------------------------------------------
// The presenter thread
// ---------------------------------------------------------------------------
// Under G.lock: every presenter submission so far may have read the slot.
void give_back(int slot) {
    G.p_used[slot] = G.p_read[slot];
    --G.slot_items[slot];
}

// DWM's refresh period and a vertical blank (QPC ticks), else the primary
// display's rate without phase. RBREW_FRAMEGEN_HZ=<rate> sets the rate.
using DwmTiming = HRESULT(WINAPI*)(HWND, DWM_TIMING_INFO*);
void display_timing(int64_t fq, int64_t& period, int64_t& vblank) {
    static DwmTiming dwm = nullptr;
    static bool tried = false;
    static double hz_env = 0.0;
    if (!tried) {
        tried = true;
        if (HMODULE m = LoadLibraryA("dwmapi.dll"))
            dwm = reinterpret_cast<DwmTiming>(reinterpret_cast<void*>(GetProcAddress(m, "DwmGetCompositionTimingInfo")));
        if (const char* e = std::getenv("RBREW_FRAMEGEN_HZ"); e && std::atof(e) >= 24.0) hz_env = std::atof(e);
    }
    DWM_TIMING_INFO ti{};
    ti.cbSize = sizeof(ti);
    if (dwm && dwm(nullptr, &ti) == S_OK && ti.qpcRefreshPeriod > 0) {
        period = int64_t(ti.qpcRefreshPeriod);
        vblank = int64_t(ti.qpcVBlank);
    } else {
        DEVMODEA dm{};
        dm.dmSize = sizeof(dm);
        const int hz = EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1
                           ? int(dm.dmDisplayFrequency)
                           : 60;
        period = fq / hz;
        vblank = 0;
    }
    if (hz_env > 0.0) period = int64_t(double(fq) / hz_env);
    if (period < fq / 1000) period = fq / 60; // nonsense from the system
}

void stat_max(std::atomic<int64_t>& m, int64_t v) {
    for (int64_t c = m.load(std::memory_order_relaxed); v > c && !m.compare_exchange_weak(c, v);) {}
}
void stat_time(std::atomic<int64_t>& sum, std::atomic<int64_t>& max, int64_t ticks) {
    const int64_t us = ticks * 1000000 / qpf();
    sum += us;
    stat_max(max, us);
}
void wait_fence(int i) {
    if (!G.p_pending[i]) return;
    vkWaitForFences(s_device, 1, &G.p_fence[i], VK_TRUE, 5000000000ull);
    vkResetFences(s_device, 1, &G.p_fence[i]);
    G.p_pending[i] = false;
}
VkSubmitInfo2 submit_info(const VkCommandBufferSubmitInfo& cb, const VkSemaphoreSubmitInfo* waits, uint32_t nw,
                          const VkSemaphoreSubmitInfo* signals, uint32_t ns) {
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = nw;
    submit.pWaitSemaphoreInfos = waits;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cb;
    submit.signalSemaphoreInfoCount = ns;
    submit.pSignalSemaphoreInfos = signals;
    return submit;
}
// Presenter: makes and presents one picture. False when nothing was shown.
// Two submissions: the compute (reads the frames, signals ptl: the render
// thread may reuse their slots once it is done, whatever the display does),
// then, after the acquire, the blit into the swapchain image.
bool show(const Pick& p) {
    const int i = G.p_i;
    G.p_i = (i + 1) % kCmds;
    wait_fence(2 * i);
    wait_fence(2 * i + 1);
    const bool between = p.mode != 1 && p.prev >= 0;
    const int o = G.out_i;
    G.out_i ^= 1;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    // --- compute: the picture into out[o]
    VkCommandBuffer cb = G.p_cmd[2 * i];
    fn.ResetCommandBuffer(cb, 0);
    fn.BeginCommandBuffer(cb, &begin);
    // out[o]: its blit two pictures ago has read it (same queue, earlier submission).
    image_barrier(fn.CmdPipelineBarrier2, cb, G.out[o].image,
                  G.out_init[o] ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    G.out_init[o] = true;
    fn.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    const bool ovl = G.has_ovl[p.cur];
    VkDescriptorImageInfo ii[4] = {
        {G.smp, G.col[between ? p.prev : p.cur].view, VK_IMAGE_LAYOUT_GENERAL},
        {G.smp, G.col[p.cur].view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, G.out[o].view, VK_IMAGE_LAYOUT_GENERAL},
        {G.smp, ovl ? G.ovl[p.cur].view : G.col[p.cur].view, VK_IMAGE_LAYOUT_GENERAL}}; // unused without overlay
    VkDescriptorBufferInfo bi{G.buf[p.cur], 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[5]{};
    for (uint32_t n = 0; n < 5; ++n) {
        wr[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[n].dstBinding = n;
        wr[n].descriptorCount = 1;
    }
    wr[0].descriptorType = wr[1].descriptorType = wr[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[0].pImageInfo = &ii[0];
    wr[1].pImageInfo = &ii[1];
    wr[4].pImageInfo = &ii[3];
    wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[2].pImageInfo = &ii[2];
    wr[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[3].pBufferInfo = &bi;
    fn.CmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pl, 0, 5, wr);
    const PushConstants pc{int32_t(G.ext.width), int32_t(G.ext.height), int32_t(G.w), int32_t(G.h),
                           int32_t(G.gw),        int32_t(G.gh),         between ? p.mode : 1, ovl ? 1 : 0,
                           G.fs,                 p.alpha};
    fn.CmdPushConstants(cb, G.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    fn.CmdDispatch(cb, (G.ext.width + 15) / 16, (G.ext.height + 7) / 8, 1);
    // Second scope: the blit submitted next on this queue.
    image_barrier(fn.CmdPipelineBarrier2, cb, G.out[o].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    fn.EndCommandBuffer(cb);
    VkSemaphoreSubmitInfo waits[2]{{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO},
                                   {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    waits[0].semaphore = s_timeline; // the frames' copies and the overlay
    waits[0].value = p.wait;
    waits[0].stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    waits[1].semaphore = G.of_tl; // the flow
    waits[1].value = p.flow_wait;
    waits[1].stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    const uint64_t pv = G.pval + 1;
    VkSemaphoreSubmitInfo done{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    done.semaphore = G.ptl;
    done.value = pv;
    done.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 submit = submit_info(cbi, waits, between ? 2 : 1, &done, 1);
    if (fn.QueueSubmit2(G.p_queue, 1, &submit, G.p_fence[2 * i]) != VK_SUCCESS) {
        G.swap_bad = true; // the render thread waits for the device and starts over
        ++G.skipped;
        return false;
    }
    G.p_pending[2 * i] = true;
    G.pval = pv;
    G.p_read[p.cur] = pv;
    if (between) G.p_read[p.prev] = pv;
    // --- the swapchain image
    VkSemaphore acquired = G.p_acq[G.p_ai];
    uint32_t index = 0;
    const int64_t t0 = qpc();
    const VkResult ar = vkAcquireNextImageKHR(s_device, G.sc, 1000000000ull, acquired, VK_NULL_HANDLE, &index);
    stat_time(G.st.acq_us, G.st.acq_max, qpc() - t0);
    if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
        if (ar != VK_TIMEOUT && ar != VK_NOT_READY) {
            G.swap_bad = true;
            ++G.st.swap_bad;
        }
        ++G.skipped;
        return false;
    }
    G.p_ai = (G.p_ai + 1) % kAcquire;
    if (index >= G.sc_count) { // cannot happen (checked at creation)
        G.swap_bad = true;
        ++G.st.swap_bad;
        ++G.skipped;
        return false;
    }
    VkImage target = G.sc_images[index];
    cb = G.p_cmd[2 * i + 1];
    fn.ResetCommandBuffer(cb, 0);
    fn.BeginCommandBuffer(cb, &begin);
    image_barrier(fn.CmdPipelineBarrier2, cb, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(G.ext.width), int32_t(G.ext.height), 1};
    fn.CmdBlitImage(cb, G.out[o].image, VK_IMAGE_LAYOUT_GENERAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                    VK_FILTER_NEAREST);
    image_barrier(fn.CmdPipelineBarrier2, cb, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_NONE, 0);
    fn.EndCommandBuffer(cb);
    VkSemaphoreSubmitInfo acq{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    acq.semaphore = acquired;
    acq.stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    VkSemaphoreSubmitInfo rendered{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    rendered.semaphore = G.p_rend[index];
    rendered.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    cbi.commandBuffer = cb;
    submit = submit_info(cbi, &acq, 1, &rendered, 1);
    if (fn.QueueSubmit2(G.p_queue, 1, &submit, G.p_fence[2 * i + 1]) != VK_SUCCESS) {
        G.swap_bad = true;
        ++G.st.swap_bad;
        ++G.skipped;
        return false;
    }
    G.p_pending[2 * i + 1] = true;
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &G.p_rend[index];
    present.swapchainCount = 1;
    present.pSwapchains = &G.sc;
    present.pImageIndices = &index;
    const int64_t t1 = qpc();
    const VkResult pr = fn.QueuePresentKHR(G.p_queue, &present);
    stat_time(G.st.pres_us, G.st.pres_max, qpc() - t1);
    // Out of date: the render thread makes the swapchain again. Suboptimal is
    // shown as it is (a size change is caught by the render thread's check);
    // treating it as out of date could rebuild the swapchain every frame.
    if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
        G.swap_bad = true;
        ++G.st.swap_bad;
        ++G.skipped;
        return true;
    }
    ++G.shown;
    ++G.st.shown;
    if (between) {
        ++G.made;
        ++G.st.made;
        ++G.st.alpha[std::clamp(int(p.alpha * 8.0f), 0, 7)];
    } else
        ++G.st.as_is;
    return true;
}

void wait_cv(CONDITION_VARIABLE& cv, DWORD ms) {
    SleepConditionVariableSRW(&cv, &G.lock, ms ? ms : 1, 0);
}

// One picture per display refresh, a lead (half a refresh) before each
// vertical blank of DWM's timing: the playhead s (in game frames: frame n at
// s = n) advances by refresh / game period every refresh and is pulled gently
// (5% a refresh; a jump when more than a frame off) toward `lag` frames
// behind the newest frame's arrival. lag must cover the GPU's time for that
// frame (drawing, copy, flow): it starts at 1.5, shrinks slowly while frames
// are ready in time and grows by 0.15 when one is not (RBREW_FRAMEGEN_LAG
// fixes it). Past the newest frame (the next one is late) the newest frame is
// held. A frame without flow (after a gap) starts over from it.
DWORD WINAPI presenter(void*) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    Held fr[kSlots]{};
    int nf = 0;
    bool have_s = false;
    int64_t seq = 0, rp = 0, vb = 0, prev_tick = 0, shown_n = -1;
    uint64_t ticks = 0;
    double s = 0.0, lag = 1.5;
    bool fixed_lag = false;
    if (const char* e = std::getenv("RBREW_FRAMEGEN_LAG"); e && std::atof(e) > 0.0) {
        fixed_lag = true;
        lag = std::atof(e);
    }
    const int64_t fq = qpf(), spin = fq / 500; // the last 2 ms before a tick: a precise sleep outside the lock
    const auto give_back_all = [&] {
        for (int i = 0; i < nf; ++i) give_back(fr[i].slot);
        nf = 0;
        have_s = false;
    };
    AcquireSRWLockExclusive(&G.lock);
    for (;;) {
        // (the lock is held here)
        if (ticks++ % 32 == 0) {
            ReleaseSRWLockExclusive(&G.lock);
            display_timing(fq, rp, vb);
            AcquireSRWLockExclusive(&G.lock);
        }
        const int64_t lead = rp / 2, now = qpc();
        const int64_t x = now + lead - vb;
        int64_t t = vb + (x >= 0 ? x / rp + 1 : 0) * rp - lead;
        while (t < prev_tick + rp / 2) t += rp;
        bool stale = false;
        for (;;) {
            if (G.stop) break;
            if (G.paused) { // pause(): everything goes back, nothing is shown
                give_back_all();
                while (G.qn) {
                    give_back(G.q[G.qh].slot);
                    G.qh = (G.qh + 1) % kQueue;
                    --G.qn;
                }
                G.held = 0;
                shown_n = -1;
                WakeAllConditionVariable(&G.idle_cv);
                wait_cv(G.cv, 100);
                stale = true;
                break;
            }
            const int64_t n2 = qpc();
            if (n2 >= t - spin) break;
            wait_cv(G.cv, DWORD((t - spin - n2) * 1000 / fq));
        }
        if (G.stop) {
            give_back_all();
            while (G.qn) {
                give_back(G.q[G.qh].slot);
                G.qh = (G.qh + 1) % kQueue;
                --G.qn;
            }
            G.held = 0;
            WakeAllConditionVariable(&G.idle_cv);
            break;
        }
        if (stale) continue;
        ReleaseSRWLockExclusive(&G.lock);
        sleep_until_qpc(t);
        AcquireSRWLockExclusive(&G.lock);
        if (G.stop || G.paused) continue; // handled above
        prev_tick = t;
        while (G.qn) { // the frames given since the last refresh
            const Item it = G.q[G.qh];
            G.qh = (G.qh + 1) % kQueue;
            --G.qn;
            if (!it.flow) { // not the neighbour of the one before
                if (nf) ++G.st.resets;
                give_back_all();
            }
            if (nf == kHold) { // the render thread's next slots stay free
                give_back(fr[0].slot);
                std::memmove(fr, fr + 1, sizeof(fr[0]) * size_t(nf - 1));
                --nf;
            }
            fr[nf++] = {it.slot, it.flow, it.wait, it.flow_wait, it.t, ++seq};
        }
        G.held = nf;
        if (!nf || G.swap_bad) {
            WakeAllConditionVariable(&G.idle_cv);
            continue;
        }
        // The playhead, for the picture shown at the coming vertical blank t + lead.
        const double per = G.shared_period * double(fq) / 1000.0;
        const Held& nw = fr[nf - 1];
        const double target = double(nw.n) + double(t + lead - nw.t) / per - lag;
        if (!have_s) {
            s = target;
            have_s = true;
        } else {
            s += double(rp) / per;
            const double e = target - s;
            s += e > 1.0 || e < -1.0 ? e : 0.05 * e;
        }
        if (s < double(fr[0].n)) s = double(fr[0].n);
        if (s > double(nw.n)) s = double(nw.n); // the next frame is late: hold the newest
        Pick pick;
        int64_t pn = -1; // a frame of the game shown as it is
        if (s >= double(nw.n) || G.debug == 1) { // RBREW_FRAMEGEN_DEBUG=1: the newest frame, as it is
            pn = nw.n;
            pick.cur = nw.slot;
            pick.wait = nw.wait;
        } else {
            const int64_t i0 = int64_t(std::floor(s));
            const Held* f0 = &fr[i0 - fr[0].n];
            const Held* f1 = f0 + 1;
            const double a = s - double(i0);
            if (f1->flow && a >= 1.0 / 128.0 && a <= 1.0 - 1.0 / 128.0) {
                pick.prev = f0->slot;
                pick.cur = f1->slot;
                pick.mode = G.debug == 2 ? 2 : G.debug == 3 ? 3 : 0;
                pick.alpha = G.debug == 2 ? 0.5f : float(a);
                pick.wait = f1->wait;
                pick.flow_wait = f1->flow_wait;
            } else {
                const Held* g = a < 0.5 ? f0 : f1;
                pn = g->n;
                pick.cur = g->slot;
                pick.wait = g->wait;
            }
        }
        const bool skip = pn >= 0 && pn == shown_n; // the same picture as at the refresh before
        G.busy = !skip;
        ReleaseSRWLockExclusive(&G.lock);
        if (!skip) {
            const bool ready = counter(s_timeline) >= pick.wait && (pick.prev < 0 || counter(G.of_tl) >= pick.flow_wait);
            if (!ready) { // not ready on the GPU in time: a later picture from now on
                ++G.late;
                ++G.st.late;
                if (!fixed_lag) lag = std::min(lag + 0.15, 1.9);
            } else if (!fixed_lag)
                lag = std::max(lag - 0.00005, 1.05);
            G.st.lag_milli = int(lag * 1000.0);
            if (show(pick)) shown_n = pn;
        }
        AcquireSRWLockExclusive(&G.lock);
        G.busy = false;
        while (nf > 1 && double(fr[0].n) < std::floor(s)) { // before the playhead: no longer read
            give_back(fr[0].slot);
            std::memmove(fr, fr + 1, sizeof(fr[0]) * size_t(nf - 1));
            --nf;
        }
        G.held = nf;
        WakeAllConditionVariable(&G.idle_cv);
    }
    ReleaseSRWLockExclusive(&G.lock);
    return 0;
}

// ---------------------------------------------------------------------------
// Render thread
// ---------------------------------------------------------------------------
// RBREW_FRAMEGEN_LOG=<file>: written directly (works with RBREW_DISABLE_LOGGING).
FILE* log_file() {
    static FILE* f = [] {
        const char* path = std::getenv("RBREW_FRAMEGEN_LOG");
        return path && *path ? std::fopen(path, "w") : nullptr;
    }();
    return f;
}
void log_line(const char* format, ...) {
    FILE* f = log_file();
    if (!f) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(f, format, args);
    va_end(args);
    std::fflush(f);
}
// Once a second (render thread): what happened since the last line.
void write_log() {
    FILE* f = log_file();
    if (!f) return;
    static int64_t start = 0, last = 0;
    const int64_t now = qpc();
    if (!last) start = last = now;
    if (now - last < qpf()) return;
    last = now;
    auto& st = G.st;
    const auto x = [](std::atomic<int64_t>& a) { return (long long)a.exchange(0); };
    const long long frames = x(st.frames), shown = x(st.shown), made = x(st.made), as_is = x(st.as_is);
    const long long late = x(st.late), skipped = x(st.skipped), dropped = x(st.dropped), resets = x(st.resets);
    const long long hook = x(st.hook_us), hook_max = x(st.hook_max), wait = x(st.wait_us), wait_max = x(st.wait_max);
    const long long timeouts = x(st.slot_timeouts), acq = x(st.acq_us), acq_max = x(st.acq_max);
    const long long pres = x(st.pres_us), pres_max = x(st.pres_max), bad = x(st.swap_bad);
    const long long fb_capture = x(st.fb_capture), fb_source = x(st.fb_source), fb_min = x(st.fb_minimized);
    const long long recreate = x(st.recreate), rebuild = x(st.rebuild), pauses = x(st.pauses);
    long long alpha[8];
    for (int i = 0; i < 8; ++i) alpha[i] = x(st.alpha[i]);
    const double per_frame = frames ? 1.0 / double(frames) : 0.0;
    const double per_pic = shown + skipped ? 1.0 / double(shown + skipped) : 0.0;
    std::fprintf(f,
                 "t=%.0f on=%d debug=%d | game frames %lld, pictures %lld (made %lld, as is %lld), late %lld, "
                 "not shown %lld, dropped %lld, flow resets %lld | render thread: present %.2f/%.2f ms, slot wait "
                 "%.2f/%.2f ms (avg/max), timeouts %lld | presenter: acquire %.2f/%.2f ms, present %.2f/%.2f ms, "
                 "swapchain bad %lld | normal presents: capture %lld, source/size %lld, minimized %lld | swapchain "
                 "rebuilds %lld, resource rebuilds %lld, pauses %lld | lag %.2f frames, period %.2f ms | alpha "
                 "%lld %lld %lld %lld %lld %lld %lld %lld | %s %u images %ux%u, frames %ux%u, flow %ux%u\n",
                 double(now - start) / double(qpf()), int(G.on), G.debug, frames, shown, made, as_is, late, skipped,
                 dropped, resets, hook * per_frame / 1000.0, hook_max / 1000.0, wait * per_frame / 1000.0,
                 wait_max / 1000.0, timeouts, acq * per_pic / 1000.0, acq_max / 1000.0, pres * per_pic / 1000.0,
                 pres_max / 1000.0, bad, fb_capture, fb_source, fb_min, recreate, rebuild, pauses,
                 st.lag_milli.load() / 1000.0, G.period, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5],
                 alpha[6], alpha[7],
                 s_swapchain.mode == VK_PRESENT_MODE_FIFO_KHR        ? "FIFO"
                 : s_swapchain.mode == VK_PRESENT_MODE_MAILBOX_KHR   ? "MAILBOX"
                 : s_swapchain.mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE"
                                                                     : "other",
                 uint32_t(s_swapchain.images.size()), s_swapchain.extent.width, s_swapchain.extent.height, G.w, G.h,
                 G.fw, G.fh);
    std::fflush(f);
}

// Until the presenter has given every frame back and does nothing; it stays
// so (touches neither the swapchain nor its queue) until the next push.
void pause() {
    if (!G.thread) return;
    ++G.st.pauses;
    AcquireSRWLockExclusive(&G.lock);
    G.paused = true;
    WakeAllConditionVariable(&G.cv);
    const int64_t until = qpc() + qpf() * 5;
    while ((G.qn || G.busy || G.held) && qpc() < until) wait_cv(G.idle_cv, 100);
    ReleaseSRWLockExclusive(&G.lock);
}
void stop_thread() {
    if (!G.thread) return;
    pause();
    AcquireSRWLockExclusive(&G.lock);
    G.stop = true;
    WakeAllConditionVariable(&G.cv);
    ReleaseSRWLockExclusive(&G.lock);
    WaitForSingleObject(G.thread, 5000);
    CloseHandle(G.thread);
    G.thread = nullptr;
    G.stop = false;
    G.paused = false;
    G.qh = G.qn = G.held = 0;
    G.busy = false;
    for (auto& n : G.slot_items) n = 0;
}
// The device is idle: nothing of the presenter, the flow or the frames runs.
void idle() {
    pause();
    vkDeviceWaitIdle(s_device);
    poll_completed();
}

void turn_off() {
    if (!G.on) return;
    stop_thread();
    vkDeviceWaitIdle(s_device);
    poll_completed();
    free_slots();
    free_screen();
    s_fg_fifo = false; // the normal present makes the swapchain again (MAILBOX / IMMEDIATE)
    G.sc = VK_NULL_HANDLE;
    G.sc_count = 0;
    G.on = false;
    RBREW_LOG_INFO("vk",
                   "frame gen: off (%llu game frames; %lld pictures shown, %lld made between frames, %lld late, "
                   "%lld not shown)\n",
                   (unsigned long long)G.frames, (long long)G.shown.load(), (long long)G.made.load(),
                   (long long)G.late.load(), (long long)G.skipped.load());
    log_line("frame gen: off (%llu game frames; %lld pictures shown, %lld made between frames, %lld late, %lld not "
             "shown)\n",
             (unsigned long long)G.frames, (long long)G.shown.load(), (long long)G.made.load(),
             (long long)G.late.load(), (long long)G.skipped.load());
}
void fail(const char* why) {
    RBREW_LOG_WARN("vk", "frame gen: %s: normal presentation\n", why);
    log_line("frame gen: FAILED: %s: normal presentation until restart\n", why);
    G.failed = true;
    if (G.on)
        turn_off();
    else {
        vkDeviceWaitIdle(s_device);
        free_slots();
        free_screen();
        s_fg_fifo = false;
    }
}
bool turn_on() {
    if (!G.dev || G.failed) return false;
    if (!G.objects && !make_objects()) {
        vkDeviceWaitIdle(s_device);
        destroy_objects();
        fail("the compute stage or its objects could not be made");
        return false;
    }
    G.stop = G.paused = G.busy = false;
    G.qh = G.qn = G.held = 0;
    for (auto& n : G.slot_items) n = 0;
    G.swap_bad = false;
    if (const char* d = std::getenv("RBREW_FRAMEGEN_DEBUG"); d && *d) G.debug = std::clamp(std::atoi(d), 0, 3);
    G.thread = CreateThread(nullptr, 0, presenter, nullptr, 0, nullptr);
    if (!G.thread) {
        fail("the presenter thread could not be started");
        return false;
    }
    G.on = true;
    G.have_prev = false;
    G.t_last = 0;
    G.period = 1000.0 / 60.0;
    G.frames = 0;
    G.shown = G.made = G.late = G.skipped = 0;
    for (auto& v : G.serials) v = 0;
    RBREW_LOG_INFO("vk", "frame gen: on (one picture per display refresh, FIFO; optical flow family %u)\n",
                   G.of_family);
    log_line("frame gen: on (optical flow family %u, presenter queue %u/1, debug %d)\n", G.of_family, s_queue_family,
             G.debug);
    return true;
}

// Slot k is no longer read by the presenter or the optical flow queue.
void wait_slot(int k) {
    uint64_t used = 0;
    AcquireSRWLockExclusive(&G.lock);
    const int64_t until = qpc() + qpf() * 2;
    while (G.slot_items[k] && qpc() < until) wait_cv(G.idle_cv, 100);
    if (G.slot_items[k]) ++G.st.slot_timeouts;
    used = G.p_used[k];
    ReleaseSRWLockExclusive(&G.lock);
    wait_value(G.ptl, used, 2000000000ull);
    for (int j = 0; j < 2; ++j) wait_value(G.of_tl, G.of_used[(k + j) % kSlots], 5000000000ull);
}

// The optical flow queue's part of slot k: flow both ways into buf[k], after
// the copy (s_timeline >= serial). Returns the of_tl value, 0 on failure.
uint64_t measure_flow(int k, uint64_t serial) {
    VkCommandBuffer cb = G.of_cmd[k];
    fn.ResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    fn.BeginCommandBuffer(cb, &begin);
    for (auto& f : G.flow) // the copy of the run before has read them (same queue)
        image_barrier(fn.CmdPipelineBarrier2, cb, f.image,
                      G.flow_init ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV,
                      VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV);
    G.flow_init = true;
    VkOpticalFlowExecuteInfoNV ei{VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV};
    ei.flags = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV; // the session ran kSlots frames ago
    fn.CmdOpticalFlowExecuteNV(cb, G.session[k], &ei);
    for (auto& f : G.flow)
        image_barrier(fn.CmdPipelineBarrier2, cb, f.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV, VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV,
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < 2; ++i) {
        VkBufferImageCopy c{};
        c.bufferOffset = VkDeviceSize(i) * G.gw * G.gh * 4;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {G.gw, G.gh, 1};
        fn.CmdCopyImageToBuffer(cb, G.flow[i].image, VK_IMAGE_LAYOUT_GENERAL, G.buf[k], 1, &c);
    }
    fn.EndCommandBuffer(cb);
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = s_timeline;
    wait.value = serial;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = G.of_tl;
    signal.value = G.of_val + 1;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cbi;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    if (fn.QueueSubmit2(G.of_queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) return 0;
    G.of_used[k] = ++G.of_val;
    return G.of_val;
}

void push(const Item& it) {
    AcquireSRWLockExclusive(&G.lock);
    const int64_t until = qpc() + qpf() * 2;
    while (G.qn == kQueue && qpc() < until) wait_cv(G.idle_cv, 100);
    if (G.qn == kQueue) { // the presenter does not take frames: drop the oldest
        ++G.st.dropped;
        --G.slot_items[G.q[G.qh].slot];
        G.qh = (G.qh + 1) % kQueue;
        --G.qn;
    }
    G.q[(G.qh + G.qn) % kQueue] = it;
    ++G.qn;
    ++G.slot_items[it.slot];
    G.paused = false;
    G.shared_period = G.period;
    WakeAllConditionVariable(&G.cv);
    ReleaseSRWLockExclusive(&G.lock);
}

// gx2vk::present: 1 = presented by frame gen, -1 = nothing to present
// (present returns false), 0 = present the normal way (the presenter is
// paused or off then).
int present_frame(Image& source, uint32_t width, uint32_t height) {
    const bool want = G.dev && !G.failed && rbrew_settings::g_frame_gen.load(std::memory_order_relaxed);
    if (!want) {
        if (G.on) turn_off();
        return 0;
    }
    if (!G.on && !turn_on()) return 0;
    // The flow is measured on frames of at most flow_lines() lines (make_slots).
    const uint32_t flow_h = std::min(height, flow_lines()), flow_w = height ? uint32_t(uint64_t(width) * flow_h / height) : 0;
    // A photo or test capture (read back, without the overlay), a black
    // picture, or a frame size the optical flow engine does not take: the
    // normal path presents it, the presenter waits.
    if (s_capture_presents || !source.image || width < G.of_min_w || height < G.of_min_h ||
        (G.of_max_w && flow_w > G.of_max_w) || (G.of_max_h && flow_h > G.of_max_h)) {
        ++(s_capture_presents ? G.st.fb_capture : G.st.fb_source);
        pause();
        return 0;
    }
    int dw = 0, dh = 0;
    SDL_Vulkan_GetDrawableSize(s_window, &dw, &dh);
    if (dw <= 0 || dh <= 0) { // minimized
        ++G.st.fb_minimized;
        pause();
        return -1;
    }
    if (!s_swapchain.swapchain || uint32_t(dw) != s_swapchain.extent.width ||
        uint32_t(dh) != s_swapchain.extent.height || !s_swapchain.framegen || G.swap_bad) {
        ++G.st.recreate;
        idle();
        s_fg_fifo = true;
        G.swap_bad = false;
        if (!create_swapchain()) return -1;
        if (s_swapchain.images.size() > size_t(kMaxImages)) {
            fail("too many swapchain images");
            return 0;
        }
    }
    if (G.sc != s_swapchain.swapchain) { // made again here or by a normal present (the presenter is paused then)
        pause();
        if (s_swapchain.images.size() > size_t(kMaxImages)) {
            fail("too many swapchain images");
            return 0;
        }
        G.sc = s_swapchain.swapchain;
        G.sc_count = uint32_t(s_swapchain.images.size());
        for (uint32_t i = 0; i < G.sc_count; ++i) G.sc_images[i] = s_swapchain.images[i];
    }
    if (G.ext.width != s_swapchain.extent.width || G.ext.height != s_swapchain.extent.height ||
        G.ovl_format != s_swapchain.format) {
        ++G.st.rebuild;
        idle();
        free_screen();
        if (!make_screen(s_swapchain.extent, s_swapchain.format)) {
            fail("out of memory for the swapchain-sized images");
            return 0;
        }
    }
    if (G.w != width || G.h != height) {
        ++G.st.rebuild;
        idle();
        free_slots();
        if (!make_slots(width, height)) {
            fail("the frame images or the optical flow sessions could not be made");
            return 0;
        }
    }
    {
        const int64_t now = qpc();
        const double ms = G.t_last ? double(now - G.t_last) * 1000.0 / double(qpf()) : 0.0;
        if (ms > 8.0 && ms < 50.0) G.period += (ms - G.period) * 0.1;
        if (ms > 70.0) G.have_prev = false; // a long gap (loading, a hold): the frame before is not its neighbour
        G.t_last = now;
    }
    const int k = G.k;
    const int64_t w0 = qpc();
    wait_slot(k);
    // Frames in flight, as the normal present bounds them (ImGui's buffers
    // rotate over the swapchain image count).
    if (const uint64_t old = G.serials[G.serial_i]) wait_serial(old);
    stat_time(G.st.wait_us, G.st.wait_max, qpc() - w0);

    VkCommandBuffer cb = cmd();
    barrier_if_dirty();
    // col[k]: last written by this queue kSlots frames ago, read by the
    // presenter and the flow since (waited for above).
    image_barrier(vkCmdPipelineBarrier2, cb, G.col[k].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
    vkCmdBlitImage(cb, source.image, VK_IMAGE_LAYOUT_GENERAL, G.col[k].image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                   VK_FILTER_NEAREST);
    if (G.reduced) { // the smaller copy the flow is measured on
        image_barrier(vkCmdPipelineBarrier2, cb, G.col[k].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        image_barrier(vkCmdPipelineBarrier2, cb, G.cols[k].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageBlit reduce{};
        reduce.srcSubresource = reduce.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reduce.srcOffsets[1] = {int32_t(width), int32_t(height), 1};
        reduce.dstOffsets[1] = {int32_t(G.fw), int32_t(G.fh), 1};
        vkCmdBlitImage(cb, G.col[k].image, VK_IMAGE_LAYOUT_GENERAL, G.cols[k].image, VK_IMAGE_LAYOUT_GENERAL, 1, &reduce,
                       VK_FILTER_LINEAR);
    }
    // The overlay (menu, FPS) of this frame, composited by the presenter.
    G.has_ovl[k] = false;
    if (gx2_imgui::ensure_ready(s_window, s_instance, s_physical, s_device, s_queue_family, s_queue,
                                s_swapchain.format, uint32_t(s_swapchain.images.size()))) {
        if (ImDrawData* overlay = gx2_imgui::begin_frame(s_swapchain.extent)) {
            const VkImage image = G.ovl[k].image;
            image_barrier(vkCmdPipelineBarrier2, cb, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                          VK_ACCESS_2_TRANSFER_WRITE_BIT);
            const VkClearColorValue clear{};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            // Leaves the image in TRANSFER_DST_OPTIMAL (also when it draws nothing).
            gx2_imgui::render_draw_data(overlay, cb, image, G.ovl[k].view, s_swapchain.extent);
            image_barrier(vkCmdPipelineBarrier2, cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            G.has_ovl[k] = true;
        }
    }
    const uint64_t serial = submit_with(nullptr, 0, nullptr);
    G.serials[G.serial_i] = serial;
    G.serial_i = (G.serial_i + 1) % 3;
    Item it{k, false, serial, 0, 0};
    if (G.have_prev) {
        it.flow_wait = measure_flow(k, serial);
        it.flow = it.flow_wait != 0;
    }
    it.t = qpc();
    push(it);
    G.have_prev = true;
    G.k = (k + 1) % kSlots;
    ++G.frames;
    ++G.st.frames;
    ++s_presents;
    poll_completed();
    return 1;
}
int present_hook(Image& source, uint32_t width, uint32_t height) {
    const int64_t t0 = qpc();
    const int result = present_frame(source, width, height);
    if (result > 0) stat_time(G.st.hook_us, G.st.hook_max, qpc() - t0);
    write_log();
    return result;
}

// gx2vk::shutdown (after the recording thread stopped, before the device goes).
void shutdown_all() {
    if (G.on) turn_off();
    stop_thread();
    if (s_device) {
        vkDeviceWaitIdle(s_device);
        free_slots();
        free_screen();
        destroy_objects();
    }
    s_fg_fifo = false;
    G.on = false;
    G.failed = false;
    G.dev = false;
    G.of_queue = G.p_queue = VK_NULL_HANDLE;
}

} // namespace fg
} // namespace
#else
namespace {
namespace fg {
inline void device_prepare(const std::vector<VkExtensionProperties>&, std::vector<const char*>&, void**&, bool) {}
inline void device_queues(VkDeviceCreateInfo&, const VkDeviceQueueCreateInfo&) {}
inline bool device_revert(std::vector<const char*>&, VkDeviceCreateInfo&, const VkDeviceQueueCreateInfo*) {
    return false;
}
inline void device_created() {}
inline int present_hook(Image&, uint32_t, uint32_t) {
    return 0;
}
inline void shutdown_all() {}
} // namespace fg
} // namespace
#endif
