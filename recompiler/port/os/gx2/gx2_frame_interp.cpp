#include "gx2_frame_interp.h"
#include "../../runtime/rebrewu_runtime.h"
#include "gx2_native_context.h"
#include "../../runtime/port_settings.h"
#include "../../runtime/benchmark_input.h"
#include "vk/gx2_vk_capture.h"
#include "../../hooks/mouse_aim.h"
#include "../../runtime/splatoon_diagnostics.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>
#include <memory>
#include <unordered_map>
#include <vector>

void gx2_render_query_poll_completed(); // gx2_render.h
void gx2_render_reset_command_tracking(); // gx2_render_vk_draw.cpp

namespace gx2interp {
namespace {
using Clock = std::chrono::steady_clock;
uint64_t now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

std::atomic<bool> s_wanted{false};

// One draw unit's uniform bytes: the packed register block and the guest
// uniform blocks, per stage (0 vertex, 1 pixel), as offsets into Record::bytes.
struct Unit {
    uint64_t key = 0, base = 0; // key = base + occurrence
    uint32_t uf_off[2]{}, uf_size[2]{};
    uint32_t bank_off[2][16]{}, bank_size[2][16]{};
    uint32_t bank_mask[2]{};
    bool has_uf[2]{};
    // Uniform buffers the draw bound (for the Vulkan replay's substitution).
    // owner: the unit whose recorded bytes the buffer holds (a draw that
    // does not set a block keeps the binding an earlier draw uploaded).
    struct Ref {
        uint64_t buffer, offset;
        uint32_t owner;
        uint8_t stage, bank;
    }; // bank 255: register block
    Ref refs[36];
    bool transparent = false; // color blending on (glass, effects)
    uint8_t ref_count = 0;
};
struct Record {
    rbrew_aim::ViewTag view; // the frame's camera and mouse counts (display-rate mouse look)
    std::vector<Unit> units;
    std::vector<uint8_t> bytes;
    std::unordered_map<uint64_t, uint32_t> by_key;
    std::unordered_map<uint64_t, uint32_t> occurrences; // per base key: how many draws
    void clear() {
        view = {};
        units.clear();
        bytes.clear();
        by_key.clear();
        occurrences.clear();
    }
    uint32_t put(const void* data, size_t size) {
        const uint32_t offset = uint32_t(bytes.size());
        bytes.insert(bytes.end(), static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        return offset;
    }
};
constexpr size_t kMaxRecordedUnits = 100000;
constexpr size_t kMaxRecordedBytes = size_t(128) << 20; // 128 MiB per-frame cap (prevents OOM / u32 overflow)
void break_capture();
Record s_records[2];
Record* s_cur = &s_records[0];                        // frame being recorded (frame N during its replay)
Record* s_last = &s_records[1];                       // previous complete frame (N-1 for the replay)
std::unordered_map<uint64_t, uint32_t> s_occurrences; // per frame (record) or replay
uint32_t s_unit = UINT32_MAX;                         // recording: index of the current unit
const Unit* s_replay_cur = nullptr;                   // replay: N's unit for this draw
const Unit* s_replay_prev = nullptr;                  // replay: N-1's matching unit

std::vector<Gx2AsyncWork*> s_works, s_replay_works;
bool s_replaying = false, s_pending_replay = false, s_frame_broken = false;
// Vulkan replay (gx2_vk_capture.h): the captured commands of frame N, issued
// again with blended uniform buffers. RBREW_INTERP_VK=0 replays the GX2
// commands instead (the first implementation, far more render-thread time).
bool s_vk_replay = false, s_capture_mid_frame = false, s_capture_armed = false;
// A failed replay (exception) stops the in-between frames for a while: the
// game runs at plain 60 instead of the process aborting (Recon at 165 Hz,
// 2026-11-21: fail-fast during a stage load).
uint64_t s_fault_until_ns = 0;
void note_replay_fault(const char* what);
uint64_t s_frame_first_serial = 0;     // recording serial when the frame's capture started
std::vector<const void*> s_prev_cache; // per unit of N: matched N-1 unit (or the sentinel)
struct SpanOwner {
    uint32_t unit;
    uint8_t stage, bank;
};
struct Substituted {
    VkBuffer buffer;
    VkDeviceSize offset;
};
std::unordered_map<uint64_t, Substituted> s_substituted; // replay: (owner, stage, bank, range) -> blended span
std::unordered_map<uint64_t, Substituted> s_by_content;  // replay: block bytes -> what the first draw with them got
std::unordered_map<uint64_t, SpanOwner> s_span_owner;    // this frame: uniform span -> unit that recorded its bytes
uint64_t span_key(uint64_t buffer, uint64_t offset) {
    return buffer * 0x9E3779B97F4A7C15ull ^ offset;
}
bool vk_mode() {
    static const bool on = [] {
        const char* v = std::getenv("RBREW_INTERP_VK");
        return !(v && v[0] == '0');
    }();
    return on;
}
unsigned s_scan_index = 0;
uint64_t s_last_complete = 0;
double s_last_complete_interval_ns = 0;
double s_interval_ns = 16683333.0;
// Render-thread time of a frame's own work and of one in-between frame
// (averages): what the controller's "does this rate fit" test uses.
uint64_t s_frame_busy_ns = 0;
double s_busy_avg_ns = 0, s_replay_avg_ns = 0;
// Render-thread time of an in-between frame per captured command: a replay
// costs about its frame's command count times this (0.5 ms in a light scene,
// 3-7 ms facing the plaza's centre). The average alone, taken in a light
// scene, let the controller raise the rate where the replays could not fit.
double s_replay_ns_per_cmd = 0;
uint64_t s_busy_last_ns = 0; // the last complete frame's own work
// The game's own frame interval: the game waits for the renderer at queries
// and at its swap, so render time alone does not tell whether the replays
// slow the game.
double s_game_interval_ns = 0;
// Per step: matrices blended and matrices kept because they jumped.
unsigned s_blended_matrices = 0, s_jumped_matrices = 0;
float s_blend_t = 0.5f; // blend position of the in-between frame being drawn (0: N-1, 1: N)

// The schedule (see gx2_frame_interp.h). Pictures wait in copies of the scan
// image (kSlots of them, used in turn) until their present time.
constexpr double kLogicFrameNs = 1e9 * 1001.0 / 60000.0;
constexpr unsigned kMaxPerStep = 8; // pictures planned for one game step
struct Shot {
    unsigned slot;
    uint64_t due;
    float t; // position between N-1 (0) and N (1); 1: the game's own frame
};
std::vector<Shot> s_shots; // waiting pictures, in present order
unsigned s_next_slot = 0;
struct Plan {
    unsigned count = 0;
    float t[kMaxPerStep]{};
    uint64_t due[kMaxPerStep]{};
    unsigned held_slot = 0; // the copy of frame N
};
Plan s_plan;
double s_rate = 1.0;          // presents per game step now (the controller's level)
unsigned s_level = 0;         // index in the controller's ladder (0: the target rate)
double s_phase = 0.0;         // position of the next picture within the step being planned
double s_phase_lo = 0.0;      // smallest position at this rate: shown without delay
double s_anchor_ns = 0.0;     // smoothed completion time of the newest frame
double s_jitter_ns = 0.0;     // how far completions stray from it
double s_lead_ns = 2.5e6;     // from a frame's completion to its first in-between picture
double s_lock_offset_ns = 0;  // fixed refresh: pull towards the display's vertical blank
uint64_t s_last_present_ns = 0, s_burst_waited_ns = 0;
bool s_schedule_reset = true; // the next planned step starts a new schedule
Status s_status;
std::vector<uint8_t> s_blend; // replay_bank result
// Replayed frames whose GPU work may still be pending (replay_may_read): the
// first serial of the captured frame and the last submission holding one of
// its replays. A few steps are enough: older ones are long complete.
struct Hold {
    uint64_t from = 0, until = 0;
};
constexpr unsigned kHolds = 16;
Hold s_holds[kHolds];
unsigned s_hold_next = 0;
// RBREW_INTERP_VB_GUARD=0 (diagnostics, the bug's reproduction): only count
// the hazards (stats "hazards"), do not orphan.
bool vb_guard() {
    static const bool on = [] {
        const char* v = std::getenv("RBREW_INTERP_VB_GUARD");
        return !(v && v[0] == '0');
    }();
    return on;
}

struct Stats {
    unsigned frames = 0, replays = 0, held = 0, between = 0, presents = 0, direct = 0, skipped = 0, dropped = 0,
             level_downs = 0, units = 0, matched = 0, missing = 0, failed = 0, changed = 0, set_binds = 0, cuts = 0,
             lock_samples = 0, hazards = 0, late1 = 0, views = 0, look = 0, views_n = 0, views_p = 0, why_capture = 0, why_broken = 0, why_units = 0,
             why_incomplete = 0, why_mid = 0;
    double look_yaw = 0;
    double replay_ms = 0, busy_ms = 0, late_ms = 0, late_max_ms = 0, lock_error_ms = 0;
};
Stats s_stats;
FILE* stats_file() {
    static FILE* f = [] {
        const char* p = std::getenv("RBREW_INTERP_STATS");
        return p && *p ? std::fopen(p, "w") : nullptr;
    }();
    return f;
}
// RBREW_INTERP_STATS=<file>: one line per second. steps: game frames;
// presents: pictures the schedule presented (between: in-between frames,
// exact: the game's own frames) plus direct: frames presented as they came
// (no in-between frame); rate: presents per step now; replay: render-thread
// time per in-between frame and its share of one core; late: how long after
// their time pictures went out.
void report_stats() {
    FILE* f = stats_file();
    if (!f) return;
    static uint64_t second = now_ns();
    const uint64_t now = now_ns();
    if (now - second < 1000000000ull) return;
    const double seconds = double(now - second) / 1e9;
    std::fprintf(f,
                 "steps %u presents %u between %u exact %u direct %u skipped %u dropped %u | rate %.3f level %u "
                 "downs %u | replay %.2f ms core %.1f%% busy %.2f ms lead %.2f ms | late avg %.2f max %.2f ms | "
                 "units %u matched %.0f%% changed %u missing %u failed %u cuts %u | lock %.2f ms | hazards %u late1 %u | look %u views %u n %u p %u yaw %.2f | why c%u b%u u%u i%u m%u\n",
                 s_stats.frames, s_stats.presents + s_stats.direct, s_stats.between, s_stats.held, s_stats.direct,
                 s_stats.skipped, s_stats.dropped, s_rate, s_level, s_stats.level_downs,
                 s_stats.replays ? s_stats.replay_ms / s_stats.replays : 0.0, s_stats.replay_ms / (10.0 * seconds),
                 s_stats.frames ? s_stats.busy_ms / s_stats.frames : 0.0, s_lead_ns / 1e6,
                 s_stats.presents ? s_stats.late_ms / s_stats.presents : 0.0, s_stats.late_max_ms, s_stats.units,
                 s_stats.units ? 100.0 * s_stats.matched / s_stats.units : 0.0, s_stats.changed, s_stats.missing,
                 s_stats.failed, s_stats.cuts, s_stats.lock_samples ? s_stats.lock_error_ms / s_stats.lock_samples : 0.0,
                 s_stats.hazards, s_stats.late1, s_stats.look, s_stats.views, s_stats.views_n, s_stats.views_p,
                 s_stats.look ? s_stats.look_yaw / s_stats.look : 0.0, s_stats.why_capture, s_stats.why_broken,
                 s_stats.why_units, s_stats.why_incomplete, s_stats.why_mid);
    std::fflush(f);
    s_stats = {};
    second = now;
}

uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h * 0xFF51AFD7ED558CCDull;
}

void drop_works(std::vector<Gx2AsyncWork*>& works) {
    for (Gx2AsyncWork* work : works) delete work;
    works.clear();
}

// A float word that blends: finite, not denormal/huge (else it is an integer
// or bit pattern), magnitude in a plausible range.
bool plausible(float v) {
    if (!std::isfinite(v)) return false;
    const float a = std::fabs(v);
    return a == 0.0f || (a > 1e-20f && a < 1e20f);
}
float to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
uint32_t to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}
uint32_t be(const uint8_t* p) {
    uint32_t w;
    std::memcpy(&w, p, 4);
    return rbrew_bswap32(w);
}
uint32_t le(const uint8_t* p) {
    uint32_t w;
    std::memcpy(&w, p, 4);
    return w;
}
// Only transforms are blended: three consecutive vec4 rows (or columns)
// whose 3x3 parts are mutually orthogonal - camera, view-projection, object
// and bone matrices - plus the vec4 after them (the translation column or
// the projection's w row) unless it starts another matrix. Every other word
// (ink and material parameters, colours, counters, packed data) stays at
// frame N: blending those turned the ground blue and hid the player.
// Guest uniform blocks are big-endian; the renderer's packed register block
// (uniform registers, remapped constants) is host order.
bool vec3_ok(const float* v, float& norm) {
    for (int i = 0; i < 4; ++i)
        if (!plausible(v[i])) return false;
    norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return norm > 1e-6f && norm < 1e7f;
}
bool is_matrix(const float* m) { // m: 3 vec4
    float n0, n1, n2;
    if (!vec3_ok(m, n0) || !vec3_ok(m + 4, n1) || !vec3_ok(m + 8, n2)) return false;
    auto ortho = [](const float* a, const float* b, float na, float nb) {
        return std::fabs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) <= 0.02f * na * nb;
    };
    return ortho(m, m + 4, n0, n1) && ortho(m, m + 8, n0, n2) && ortho(m + 4, m + 8, n1, n2);
}
bool vec4_ok(const float* v) {
    for (int i = 0; i < 4; ++i)
        if (!plausible(v[i])) return false;
    return true;
}
bool parallel(const float* x, const float* y) {
    const float nx = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]),
                ny = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
    if (nx < 1e-6f || ny < 1e-6f) return false;
    return std::fabs(x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) >= 0.98f * nx * ny;
}
// Not motion between two frames: a row turning by more than ~26 degrees
// (another object, a camera cut) or a translation changing by more than a
// quarter of its size (a texture matrix switching buffers, a teleport).
bool transform_jump(const float* a, const float* b) {
    for (int row = 0; row < 3; ++row) {
        const float* x = a + row * 4;
        const float* y = b + row * 4;
        const float dx = x[0] - y[0], dy = x[1] - y[1], dz = x[2] - y[2];
        const float diff = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float scale = std::max(std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]),
                                     std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]));
        if (diff > 0.45f * scale) return true;
    }
    const float dt =
        std::sqrt((a[3] - b[3]) * (a[3] - b[3]) + (a[7] - b[7]) * (a[7] - b[7]) + (a[11] - b[11]) * (a[11] - b[11]));
    const float ta = std::sqrt(a[3] * a[3] + a[7] * a[7] + a[11] * a[11]),
                tb = std::sqrt(b[3] * b[3] + b[7] * b[7] + b[11] * b[11]);
    return dt > 0.05f && dt > 0.25f * std::max(ta, tb);
}
void blend_bytes(const uint8_t* prev, const uint8_t* cur, uint8_t* out, size_t size, bool host_order) {
    std::memcpy(out, cur, size);
    if (std::memcmp(prev, cur, size) == 0) return;
    // Only rows (vec4) that differ can start or belong to a moving matrix:
    // the 4 KB register block of a draw usually differs in a few rows.
    const size_t rows = size / 16;
    auto read = [&](const uint8_t* p) { return to_float(host_order ? le(p) : be(p)); };
    auto load = [&](const uint8_t* base, size_t row, size_t count, float* dst) {
        for (size_t i = 0; i < count * 4; ++i) dst[i] = read(base + row * 16 + i * 4);
    };
    auto put = [&](size_t word, float v) {
        const uint32_t w = host_order ? to_bits(v) : rbrew_bswap32(to_bits(v));
        std::memcpy(out + word * 4, &w, 4);
    };
    for (size_t r = 0; r < rows;) {
        if (std::memcmp(prev + r * 16, cur + r * 16, 16) == 0) {
            ++r;
            continue;
        }
        size_t next = r + 1;
        for (size_t start = r >= 3 ? r - 3 : 0; start <= r && start + 3 <= rows; ++start) {
            float a[16], b[16];
            const size_t avail = std::min<size_t>(4, rows - start);
            load(prev, start, avail, a);
            load(cur, start, avail, b);
            if (!is_matrix(a) || !is_matrix(b)) continue;
            // The 4th vec4 belongs to it only as a projection's w row, which is
            // parallel to the 3rd row. In the register block the vec4 after a
            // 3x4 matrix is another parameter (ink, material): blending those
            // made the ground ink flicker.
            size_t n = 12;
            if (avail == 4 && vec4_ok(a + 12) && vec4_ok(b + 12) && parallel(a + 8, a + 12) && parallel(b + 8, b + 12))
                n = 16;
            if (transform_jump(a, b)) {
                ++s_jumped_matrices;
                next = std::max(next, start + n / 4);
                break;
            } // keep frame N
            ++s_blended_matrices;
            for (size_t k = 0; k < n; ++k)
                if (a[k] != b[k]) put(start * 4 + k, a[k] + (b[k] - a[k]) * s_blend_t);
            next = std::max(next, start + n / 4);
            break;
        }
        r = next;
    }
}
// How different two recordings of a draw are (sum of float differences,
// capped per word): picks, among the previous frame's draws of the same
// program and mesh, the one that is the same object.
double distance(const Record& ra, const Unit& a, const Record& rb, const Unit& b) {
    double total = 0;
    auto add = [&](const uint8_t* pa, const uint8_t* pb, size_t size, bool host_order) {
        size = std::min<size_t>(size, 256);
        for (size_t i = 0; i + 4 <= size; i += 4) {
            const uint32_t wa = host_order ? le(pa + i) : be(pa + i), wb = host_order ? le(pb + i) : be(pb + i);
            if (wa == wb) continue;
            const float fa = to_float(wa), fb = to_float(wb);
            total += plausible(fa) && plausible(fb) ? std::min(1000.0, double(std::fabs(fa - fb))) : 1.0;
        }
    };
    for (unsigned stage = 0; stage < 2; ++stage) {
        if (a.has_uf[stage] && b.has_uf[stage] && a.uf_size[stage] == b.uf_size[stage])
            add(ra.bytes.data() + a.uf_off[stage], rb.bytes.data() + b.uf_off[stage], a.uf_size[stage], true);
        for (uint32_t m = a.bank_mask[stage] & b.bank_mask[stage]; m; m &= m - 1) {
            const unsigned bank = unsigned(std::countr_zero(m));
            if (a.bank_size[stage][bank] == b.bank_size[stage][bank])
                add(ra.bytes.data() + a.bank_off[stage][bank], rb.bytes.data() + b.bank_off[stage][bank],
                    a.bank_size[stage][bank], false);
        }
    }
    return total;
}
bool no_blend() { // RBREW_INTERP_NOBLEND=1 (diagnostics): replay frame N exactly
    static const bool on = [] {
        const char* v = std::getenv("RBREW_INTERP_NOBLEND");
        return v && v[0] == '1';
    }();
    return on;
}
// N-1's unit for N's unit `cur`: one previous draw of the same program and
// mesh is that one; several (NPCs, ink drops with one mesh, whose order
// changes with sorting) the most similar; very many (particles) none.
const Unit* match_prev(const Unit& cur) {
    if (no_blend()) return nullptr;
    // RBREW_INTERP_KEEP_TRANSPARENT=1 (testing): transparent draws keep frame N.
    static const bool keep_transparent = [] {
        const char* v = std::getenv("RBREW_INTERP_KEEP_TRANSPARENT");
        return v && v[0] == '1';
    }();
    if (cur.transparent && keep_transparent) return nullptr;
    const auto prev = s_last->occurrences.find(cur.base);
    if (prev == s_last->occurrences.end()) return nullptr;
    const uint32_t count = prev->second;
    auto unit_of = [&](uint32_t occurrence) -> const Unit* {
        const auto it = s_last->by_key.find(mix(cur.base, occurrence));
        return it != s_last->by_key.end() ? &s_last->units[it->second] : nullptr;
    };
    if (count == 1) return unit_of(1);
    // RBREW_INTERP_UNIQUE_ONLY=1 (diagnostics): never blend draws with copies.
    static const bool unique_only = [] {
        const char* v = std::getenv("RBREW_INTERP_UNIQUE_ONLY");
        return v && v[0] == '1';
    }();
    if (count > 48 || unique_only) return nullptr;
    // Draw order is mostly stable: the same occurrence, when it is close.
    const Unit* best_unit = nullptr;
    double best = 1e300;
    for (uint32_t occurrence = 1; occurrence <= count; ++occurrence) {
        if (mix(cur.base, occurrence) != cur.key) continue;
        if (const Unit* same = unit_of(occurrence)) {
            best = distance(*s_cur, cur, *s_last, *same);
            best_unit = same;
            if (best < 2.0) return same;
        }
        break;
    }
    // Many copies (text glyphs, ink drops, particles, culled props): only an
    // unchanged position in the order counts (above); never blend two different ones.
    if (count > 8) return best < 2.0 ? best_unit : nullptr;
    for (uint32_t occurrence = 1; occurrence <= count && best > 0; ++occurrence)
        if (const Unit* candidate = unit_of(occurrence); candidate && candidate != best_unit) {
            const double d = distance(*s_cur, cur, *s_last, *candidate);
            if (d < best) {
                best = d;
                best_unit = candidate;
            }
        }
    return best_unit;
}
// ---- Mouse look at the display's rate ---------------------------------------
// The game turns its camera once per step; an in-between frame shows the
// camera blended between N-1 and N. With the mouse turning the camera
// (hooks/mouse_aim.h), each in-between frame adds the mouse movement that
// came after the blended position: every count received so far minus the
// counts in the blend (N-1's plus t of N's). The game turns the camera around
// its look-at point (the eye orbits the player), so the extra yaw (about the
// world's up axis) and pitch (about the camera's horizontal axis) turn the
// world-to-view matrices around that point: M' = M * W^-1, W = the orbit. At
// the next step the game's own camera has taken those counts: nothing jumps
// back and nothing is counted twice. Only the blended scene passes are
// changed; the HUD and the crosshair stay as in frame N.
//
// The view matrices are found in frame N's uniform data by their rotation:
// three rows whose 3x3 part, against the camera basis of N's tag, is a
// view (identity up to signs) or view-projection ([sx 0 cx; 0 sy cy; 0 0 a])
// matrix; an optional fourth row along the view axis (the projection's w
// row) goes with them. RBREW_INTERP_MOUSE=0: off.
struct ViewFix {
    bool on = false;
    bool cur_on = false;
    float rot[9]{};                  // extra rotation from blended pose (bx) to tx (row-major)
    float pivot[3]{};                // blended pivot
    float rot_cur[9]{};              // extra rotation from frame-N pose (cur) to tx (for unblended draws)
    float pivot_cur[3]{};            // frame-N pivot
    float basis[9]{};                // frame N's camera basis rows: right, up, forward
    float basis_prev[9]{};           // the same from the newest tag
    float yaw = 0.0f, pitch = 0.0f;  // degrees (stats)
};
ViewFix s_fix;
// The tags of the frames being completed. A frame is drawn with the camera
// of the update before it: the pose and counts taken at swap N are the ones
// frame N+1 shows (with fast mouse turns frame N's view matrices matched the
// pose taken at swap N-1, never the one taken at swap N). So frame N gets
// the tag taken at the previous swap.
rbrew_aim::ViewTag s_incoming_view, s_held_view;
bool mouse_look_on() {
    static const bool on = [] {
        const char* v = std::getenv("RBREW_INTERP_MOUSE");
        return !(v && v[0] == '0');
    }();
    return on;
}
void normalize3(float* v) {
    const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 1e-12f)
        for (int i = 0; i < 3; ++i) v[i] /= n;
}
void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// The extra rotation of the in-between frame at position t (render thread,
// before its replay). Samples the 1000 Hz raw-mouse history monotonically
// from cur.taken_ns (Swap N-1, what frame N rendered) to s_held_view.taken_ns
// (Swap N, one full frame newer and already received): zero backwards snap on
// t=1.0, zero 0.8 ms burst duplicates, and 16.7 ms lower mouse input delay.
void prepare_view_fix(float t) {
    s_fix.on = false;
    s_fix.cur_on = false;
    const rbrew_aim::ViewTag& cur = s_cur->view;
    const rbrew_aim::ViewTag& last = s_last->view;
    if (!mouse_look_on() || !cur.active || !last.active || cur.degrees_per_count == 0.0f ||
        !std::isfinite(cur.degrees_per_count) || !std::isfinite(cur.pitch) || !std::isfinite(last.pitch) ||
        !std::isfinite(cur.pitch_min) || !std::isfinite(cur.pitch_max) || cur.pitch_min > cur.pitch_max)
        return;
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(cur.eye[i]) || !std::isfinite(cur.at[i]) || !std::isfinite(last.eye[i]) ||
            !std::isfinite(last.at[i]))
            return;
    double tx = 0.0, ty = 0.0;
    const double u = std::clamp(double(t), 0.0, 1.0);
    if (s_held_view.active && s_held_view.taken_ns > cur.taken_ns && cur.taken_ns > 0) {
        const uint64_t q = cur.taken_ns + uint64_t(u * double(s_held_view.taken_ns - cur.taken_ns));
        if (!rbrew_aim::counts_at(q, tx, ty)) return;
    } else if (s_held_view.active) {
        tx = double(cur.taken_x) + u * double(s_held_view.taken_x - cur.taken_x);
        ty = double(cur.taken_y) + u * double(s_held_view.taken_y - cur.taken_y);
    } else {
        return;
    }
    const double bx = double(last.taken_x) + u * double(cur.taken_x - last.taken_x);
    const double by = double(last.taken_y) + u * double(cur.taken_y - last.taken_y);
    float yaw = float(-(tx - bx) * cur.degrees_per_count);
    float pitch = float(-(ty - by) * cur.degrees_per_count);
    const float base_pitch = last.pitch + (cur.pitch - last.pitch) * float(u);
    pitch = std::clamp(base_pitch + pitch, cur.pitch_min, cur.pitch_max) - base_pitch;
    yaw = std::clamp(yaw, -45.0f, 45.0f);
    float yaw_cur = float(-(tx - double(cur.taken_x)) * cur.degrees_per_count);
    float pitch_cur = float(-(ty - double(cur.taken_y)) * cur.degrees_per_count);
    pitch_cur = std::clamp(cur.pitch + pitch_cur, cur.pitch_min, cur.pitch_max) - cur.pitch;
    yaw_cur = std::clamp(yaw_cur, -45.0f, 45.0f);
    const bool have_blend = std::fabs(yaw) >= 1e-4f || std::fabs(pitch) >= 1e-4f;
    const bool have_cur = std::fabs(yaw_cur) >= 1e-4f || std::fabs(pitch_cur) >= 1e-4f;
    if (!have_blend && !have_cur) return;
    const float up[3] = {0, 1, 0};
    auto build_rot = [&](const float eye_p[3], const float at_p[3], float y_deg, float p_deg, float out_rot[9]) {
        float f[3] = {at_p[0] - eye_p[0], at_p[1] - eye_p[1], at_p[2] - eye_p[2]};
        normalize3(f);
        // Yaw about +Y (as the game: atan2(x, z) grows), then pitch about the
        // horizontal axis k = f' x Y (positive looks up).
        const float a = y_deg * (3.14159265358979f / 180.0f), ca = std::cos(a), sa = std::sin(a);
        const float ry[9] = {ca, 0, sa, 0, 1, 0, -sa, 0, ca};
        float f1[3] = {ry[0] * f[0] + ry[1] * f[1] + ry[2] * f[2], ry[3] * f[0] + ry[4] * f[1] + ry[5] * f[2],
                       ry[6] * f[0] + ry[7] * f[1] + ry[8] * f[2]};
        float k[3];
        cross3(f1, up, k);
        normalize3(k);
        const float b = p_deg * (3.14159265358979f / 180.0f), cb = std::cos(b), sb = std::sin(b);
        // Rodrigues: R = cI + s[k]x + (1-c) k k^T
        const float rk[9] = {cb + (1 - cb) * k[0] * k[0],        (1 - cb) * k[0] * k[1] - sb * k[2],
                             (1 - cb) * k[0] * k[2] + sb * k[1], (1 - cb) * k[1] * k[0] + sb * k[2],
                             cb + (1 - cb) * k[1] * k[1],        (1 - cb) * k[1] * k[2] - sb * k[0],
                             (1 - cb) * k[2] * k[0] - sb * k[1], (1 - cb) * k[2] * k[1] + sb * k[0],
                             cb + (1 - cb) * k[2] * k[2]};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                out_rot[r * 3 + c] =
                    rk[r * 3 + 0] * ry[0 * 3 + c] + rk[r * 3 + 1] * ry[1 * 3 + c] + rk[r * 3 + 2] * ry[2 * 3 + c];
    };
    float eye[3], at[3];
    for (int i = 0; i < 3; ++i) {
        eye[i] = last.eye[i] + (cur.eye[i] - last.eye[i]) * t;
        at[i] = last.at[i] + (cur.at[i] - last.at[i]) * t;
    }
    if (have_blend) {
        build_rot(eye, at, yaw, pitch, s_fix.rot);
        std::copy(at, at + 3, s_fix.pivot);
    } else {
        const float id[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::copy(id, id + 9, s_fix.rot);
        std::copy(at, at + 3, s_fix.pivot);
    }
    if (have_cur) {
        build_rot(cur.eye, cur.at, yaw_cur, pitch_cur, s_fix.rot_cur);
        std::copy(cur.at, cur.at + 3, s_fix.pivot_cur);
        s_fix.cur_on = true;
    }
    // The camera bases of the tags (detection against N's own data).
    auto basis_of = [&](const rbrew_aim::ViewTag& tag, float* out) {
        float fn[3] = {tag.at[0] - tag.eye[0], tag.at[1] - tag.eye[1], tag.at[2] - tag.eye[2]};
        normalize3(fn);
        float rn[3], un[3];
        cross3(fn, up, rn);
        normalize3(rn);
        cross3(rn, fn, un);
        std::copy(rn, rn + 3, out);
        std::copy(un, un + 3, out + 3);
        std::copy(fn, fn + 3, out + 6);
    };
    basis_of(cur, s_fix.basis);
    // Also accepted: the newest pose (the one-update shift above does not
    // hold in every scene; both turn the same way).
    if (s_held_view.active)
        basis_of(s_held_view, s_fix.basis_prev);
    else
        basis_of(cur, s_fix.basis_prev);
    s_fix.yaw = yaw;
    s_fix.pitch = pitch;
    s_fix.on = have_blend;
}
// Rows m (3 vec4) of frame N: a world-to-view (or view-projection) matrix of
// N's camera.
bool is_view_matrix(const float* m, const float* bs) {
    float q[3][3];
    float n[3];
    for (int i = 0; i < 3; ++i) {
        n[i] = std::sqrt(dot3(m + 4 * i, m + 4 * i));
        if (!(n[i] > 1e-6f) || !std::isfinite(n[i])) return false;
        for (int j = 0; j < 3; ++j) q[i][j] = dot3(m + 4 * i, bs + 3 * j) / n[i];
    }
    constexpr float kOff = 0.035f, kOn = 0.3f;
    // RBREW_INTERP_VIEW_LOG=<file> (diagnostics): candidates of some pictures.
    static FILE* log = [] {
        const char* p = std::getenv("RBREW_INTERP_VIEW_LOG");
        return p && *p ? std::fopen(p, "w") : nullptr;
    }();
    if (log && (s_stats.look % 50) == 1) {
        std::fprintf(log, "yaw %.3f pitch %.3f basis f %.4f %.4f %.4f | q %.4f %.4f %.4f / %.4f %.4f %.4f / %.4f %.4f %.4f\n",
                     s_fix.yaw, s_fix.pitch, bs[6], bs[7], bs[8], q[0][0], q[0][1], q[0][2], q[1][0], q[1][1], q[1][2],
                     q[2][0], q[2][1], q[2][2]);
        std::fflush(log);
    }
    return std::fabs(q[0][1]) < kOff && std::fabs(q[1][0]) < kOff && std::fabs(q[2][0]) < kOff &&
           std::fabs(q[2][1]) < kOff && std::fabs(q[0][0]) > kOn && std::fabs(q[1][1]) > kOn &&
           std::fabs(q[2][2]) > 0.99f;
}
// Applies the extra rotation to the view matrices of a uniform block: `cur`
// is frame N's bytes (detection), `out` the in-between frame's bytes.
// When `from_cur` is true (block was not blended from N-1), rotates from frame
// N's own pose (`rot_cur`) instead of the blended pose (`rot`) so unblended
// draws never overshoot the rest of the scene.
unsigned apply_view_fix(const uint8_t* cur, uint8_t* out, size_t size, bool host_order, bool from_cur = false) {
    const size_t rows = size / 16;
    if (rows < 3) return 0;
    auto read = [&](const uint8_t* p) { return to_float(host_order ? le(p) : be(p)); };
    auto put = [&](size_t word, float v) {
        const uint32_t w = host_order ? to_bits(v) : rbrew_bswap32(to_bits(v));
        std::memcpy(out + word * 4, &w, 4);
    };
    const float* rot = from_cur ? s_fix.rot_cur : s_fix.rot;
    const float* p = from_cur ? s_fix.pivot_cur : s_fix.pivot;
    unsigned fixed = 0;
    for (size_t r = 0; r + 3 <= rows;) {
        // Cheap filter: the third row lies along the view axis.
        float z[4];
        for (int i = 0; i < 4; ++i) z[i] = read(cur + (r + 2) * 16 + i * 4);
        const float zn = std::sqrt(dot3(z, z));
        if (!(zn > 1e-6f) || (std::fabs(dot3(z, s_fix.basis + 6)) < 0.99f * zn &&
                              std::fabs(dot3(z, s_fix.basis_prev + 6)) < 0.99f * zn)) {
            ++r;
            continue;
        }
        float m[16];
        for (int i = 0; i < 12; ++i) m[i] = read(cur + r * 16 + i * 4);
        if (!vec4_ok(m) || !vec4_ok(m + 4) || !vec4_ok(m + 8)) {
            ++r;
            continue;
        }
        const float* bs = s_fix.basis;
        if (is_view_matrix(m, s_fix.basis))
            ++s_stats.views_n;
        else if (is_view_matrix(m, s_fix.basis_prev)) {
            ++s_stats.views_p;
            bs = s_fix.basis_prev;
        } else {
            ++r;
            continue;
        }
        size_t n = 3;
        if (r + 4 <= rows) {
            for (int i = 0; i < 4; ++i) m[12 + i] = read(cur + (r + 3) * 16 + i * 4);
            const float wn = std::sqrt(dot3(m + 12, m + 12));
            if (vec4_ok(m + 12) && wn > 1e-6f && std::fabs(dot3(m + 12, bs + 6)) > 0.99f * wn) n = 4;
        }
        // Pure 3x3 direction/normal/sky-rotation matrices have zero translation
        // column: rotate their basis without injecting a world-pivot offset.
        const bool has_translation = !(read(cur + (r + 0) * 16 + 12) == 0.0f && read(cur + (r + 1) * 16 + 12) == 0.0f &&
                                       read(cur + (r + 2) * 16 + 12) == 0.0f);
        for (size_t i = 0; i < n; ++i) {
            float row[4];
            for (int c = 0; c < 4; ++c) row[c] = read(out + (r + i) * 16 + c * 4);
            // row' . x = (R row) . x + row.w + row.p - (R row).p
            float nr[3];
            for (int c = 0; c < 3; ++c) nr[c] = rot[c * 3 + 0] * row[0] + rot[c * 3 + 1] * row[1] + rot[c * 3 + 2] * row[2];
            for (int c = 0; c < 3; ++c) put((r + i) * 4 + c, nr[c]);
            if (has_translation) {
                const float w = row[3] + dot3(row, p) - dot3(nr, p);
                put((r + i) * 4 + 3, w);
            }
        }
        ++fixed;
        r += n;
    }
    return fixed;
}

// Vulkan replay: a uniform buffer bound by N's draw `unit` is replaced by its
// N-1/N blend, uploaded for the in-between frame.
bool substitute(uint32_t unit, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range, VkBuffer& out_buffer,
                VkDeviceSize& out_offset) {
    const bool no_depth = (unit & gx2vk::capture::kNoDepthTest) != 0;
    if (no_depth) return false; // 2D / HUD / screen-space draws always keep exact frame N
    unit &= ~gx2vk::capture::kNoDepthTest;
    if (unit >= s_cur->units.size()) return false;
    const Unit& binder = s_cur->units[unit];
    const uint64_t handle = uint64_t(reinterpret_cast<uintptr_t>(buffer));
    const Unit::Ref* ref = nullptr;
    for (uint8_t i = 0; i < binder.ref_count; ++i)
        if (binder.refs[i].buffer == handle && binder.refs[i].offset == offset) {
            ref = &binder.refs[i];
            break;
        }
    if (!ref || ref->owner >= s_cur->units.size()) return false;
    const Unit& cur = s_cur->units[ref->owner];
    // Draws that share a span share its blend.
    const uint64_t cache_key =
        (uint64_t(ref->owner) << 40) ^ (uint64_t(ref->stage) << 36) ^ (uint64_t(ref->bank) << 28) ^ uint64_t(range);
    if (const auto hit = s_substituted.find(cache_key); hit != s_substituted.end()) {
        if (!hit->second.buffer) return false;
        out_buffer = hit->second.buffer;
        out_offset = hit->second.offset;
        return true;
    }
    s_substituted[cache_key] = {VK_NULL_HANDLE, 0}; // "no substitution" unless replaced below
    // The pre-pass and the main pass draw an object with the same uniform
    // bytes. Whatever the first draw with these bytes got (blended or frame
    // N), every later one gets too: matched separately, one pass could blend
    // an object the other kept, and the main pass's depth test against the
    // pre-pass then left holes (a green floor). RBREW_INTERP_CONTENT_CACHE=0:
    // per draw.
    static const bool content_cache = [] {
        const char* v = std::getenv("RBREW_INTERP_CONTENT_CACHE");
        return !(v && v[0] == '0');
    }();
    uint64_t content_key = 0;
    if (content_cache) {
        const uint8_t* bytes = nullptr;
        size_t length = 0;
        const unsigned st = ref->stage;
        if (ref->bank == 255) {
            if (cur.has_uf[st] && size_t(cur.uf_off[st]) + cur.uf_size[st] <= s_cur->bytes.size()) {
                bytes = s_cur->bytes.data() + cur.uf_off[st];
                length = cur.uf_size[st];
            }
        } else if (ref->bank < 16 && (cur.bank_mask[st] >> ref->bank & 1) &&
                   size_t(cur.bank_off[st][ref->bank]) + cur.bank_size[st][ref->bank] <= s_cur->bytes.size()) {
            bytes = s_cur->bytes.data() + cur.bank_off[st][ref->bank];
            length = cur.bank_size[st][ref->bank];
        }
        if (bytes) {
            uint64_t h = 1469598103934665603ull ^ length ^ (uint64_t(ref->bank == 255) << 63);
            for (size_t i = 0; i + 8 <= length; i += 8) {
                uint64_t w;
                std::memcpy(&w, bytes + i, 8);
                h = (h ^ w) * 1099511628211ull;
            }
            for (size_t i = length & ~size_t(7); i < length; ++i) h = (h ^ bytes[i]) * 1099511628211ull;
            content_key = h ^ uint64_t(range) * 0x9E3779B97F4A7C15ull;
            if (const auto hit = s_by_content.find(content_key); hit != s_by_content.end()) {
                s_substituted[cache_key] = hit->second;
                if (!hit->second.buffer) return false;
                out_buffer = hit->second.buffer;
                out_offset = hit->second.offset;
                return true;
            }
            s_by_content[content_key] = {VK_NULL_HANDLE, 0};
        }
    }
    if (s_prev_cache.size() != s_cur->units.size()) s_prev_cache.assign(s_cur->units.size(), &s_prev_cache);
    if (s_prev_cache[ref->owner] == &s_prev_cache) s_prev_cache[ref->owner] = match_prev(cur);
    const Unit* prev = static_cast<const Unit*>(s_prev_cache[ref->owner]);
    ++s_stats.units;
    // Without a match (or with the block unchanged since N-1) frame N's own
    // data is already bound - unless the mouse look turns the view.
    if (!prev && !s_fix.on && !s_fix.cur_on) return false;
    const unsigned stage = ref->stage;
    if (stage > 1) return false;
    const uint8_t* a = nullptr;
    const uint8_t* b = nullptr;
    size_t size = 0;
    if (ref->bank == 255) {
        if (!cur.has_uf[stage] || size_t(cur.uf_off[stage]) + cur.uf_size[stage] > s_cur->bytes.size()) return false;
        size = cur.uf_size[stage];
        b = s_cur->bytes.data() + cur.uf_off[stage];
        if (prev && prev->has_uf[stage] && prev->uf_size[stage] == size &&
            size_t(prev->uf_off[stage]) + size <= s_last->bytes.size())
            a = s_last->bytes.data() + prev->uf_off[stage];
    } else {
        const unsigned bank = ref->bank;
        if (bank > 15 || !(cur.bank_mask[stage] >> bank & 1) ||
            size_t(cur.bank_off[stage][bank]) + cur.bank_size[stage][bank] > s_cur->bytes.size())
            return false;
        size = cur.bank_size[stage][bank];
        b = s_cur->bytes.data() + cur.bank_off[stage][bank];
        if (prev && (prev->bank_mask[stage] >> bank & 1) && prev->bank_size[stage][bank] == size &&
            size_t(prev->bank_off[stage][bank]) + size <= s_last->bytes.size())
            a = s_last->bytes.data() + prev->bank_off[stage][bank];
    }
    if (a && std::memcmp(a, b, size) == 0) a = nullptr; // unchanged
    const bool did_blend = a && s_blend_t < 1.0f;
    const bool use_cur_fix = !did_blend && s_blend_t < 1.0f;
    const bool want_fix = use_cur_fix ? s_fix.cur_on : s_fix.on;
    if (!did_blend && !want_fix) return false; // keep N's buffer
    // Blend in host memory: the upload span is write-combined, reading it
    // back (a comparison) cost ~25 us per block.
    thread_local std::vector<uint8_t> blended;
    blended.resize(size);
    if (did_blend)
        blend_bytes(a, b, blended.data(), size, ref->bank == 255);
    else
        std::memcpy(blended.data(), b, size);
    if (want_fix) s_stats.views += apply_view_fix(b, blended.data(), size, ref->bank == 255, use_cur_fix);
    if (std::memcmp(blended.data(), b, size) == 0) return false; // nothing moved: N's buffer
    ++s_stats.changed;
    const VkDeviceSize bytes = std::max<VkDeviceSize>(range, size);
    const VkDeviceSize align = std::max<VkDeviceSize>(gx2vk::caps().limits.minUniformBufferOffsetAlignment, 16);
    const gx2vk::Span span = gx2vk::upload(bytes, align);
    if (!span.data) return false;
    std::memcpy(span.data, blended.data(), size);
    // Bindings often declare 64 KB for a block of a few hundred bytes; zeroing
    // all of it was 70 MB per in-between frame. The game's block is all the
    // shader reads; a short zeroed tail covers vec4 over-reads.
    if (bytes > size) std::memset(span.data + size, 0, size_t(std::min<VkDeviceSize>(bytes - size, 1024)));
    ++s_stats.matched;
    out_buffer = span.buffer;
    out_offset = span.offset;
    s_substituted[cache_key] = {span.buffer, span.offset};
    if (content_key) s_by_content[content_key] = {span.buffer, span.offset};
    return true;
}
// Abandon the current frame's capture immediately (e.g. stage load burst or
// allocation cap) so upload chunks and deferred resources are never pinned.
void break_capture() {
    s_frame_broken = true;
    s_unit = UINT32_MAX;
    if (gx2vk::capture::enabled()) {
        gx2vk::capture::set_enabled(false);
        gx2vk::capture::discard();
    }
    gx2vk::set_capture_hold(0);
    drop_works(s_works);
    s_cur->clear();
    s_span_owner.clear();
}
// Capture follows the frame: on while a frame is recorded for interpolation.
void sync_capture() {
    if (!vk_mode() || s_pending_replay) return;
    const bool want = capturing();
    if (want && !gx2vk::capture::enabled()) {
        gx2vk::capture::install();
        gx2vk::capture::set_enabled(true);
        // Reset redundant-state tracking so the captured frame emits its own
        // initial Vulkan state and replays cleanly even in a fresh command buffer.
        gx2_render_reset_command_tracking();
        s_capture_mid_frame = !s_capture_armed; // false: the frame starts here
        s_frame_first_serial = gx2vk::recording_serial();
        gx2vk::set_capture_hold(s_frame_first_serial);
        s_capture_armed = false;
    } else if (!want && gx2vk::capture::enabled()) {
        gx2vk::capture::set_enabled(false);
        gx2vk::capture::discard();
        gx2vk::set_capture_hold(0);
    }
}
} // namespace

// ESC menu "120 FPS mode" (settings.ini fps120; RBREW_FPS120=interp turns it on).
// Off while frame gen (optical flow, gx2_vk_framegen.inl) makes the pictures
// between frames: the draw replay would add presents of its own.
bool mode_enabled() {
    return rbrew_settings::g_fps120.load(std::memory_order_relaxed) &&
           !rbrew_settings::g_frame_gen.load(std::memory_order_relaxed);
}
void set_wanted(bool wanted) {
    s_wanted.store(wanted && mode_enabled(), std::memory_order_relaxed);
}
bool capturing() {
    return !s_replaying && !s_pending_replay && !s_frame_broken && s_wanted.load(std::memory_order_relaxed);
}
bool replaying() {
    return s_replaying;
}
bool replay_draw_known() {
    return s_replay_cur != nullptr;
}
void add_busy(uint64_t ns) {
    if (!s_replaying) s_frame_busy_ns += ns;
}

void note_ubo(unsigned stage, unsigned bank, uint64_t buffer, uint64_t offset) {
    if (s_unit == UINT32_MAX || s_unit >= s_cur->units.size() || !capturing() || stage > 1) return;
    Unit& unit = s_cur->units[s_unit];
    if (unit.ref_count >= sizeof(unit.refs) / sizeof(unit.refs[0])) return;
    const bool own = bank == 255 ? unit.has_uf[stage] : bank < 16 && (unit.bank_mask[stage] >> bank & 1);
    const uint64_t key = span_key(buffer, offset);
    SpanOwner owner{s_unit, uint8_t(stage), uint8_t(bank)};
    try {
        if (own)
            s_span_owner.try_emplace(key, owner);
        else if (const auto it = s_span_owner.find(key); it != s_span_owner.end())
            owner = it->second;
        else
            return; // bound before this frame's capture: no bytes to blend
    } catch (const std::exception& error) {
        break_capture();
        note_replay_fault(error.what());
        return;
    } catch (...) {
        break_capture();
        note_replay_fault("note_ubo exception");
        return;
    }
    unit.refs[unit.ref_count++] = {buffer, offset, owner.unit, owner.stage, owner.bank};
}
void before_item() {
    try {
        sync_capture();
    } catch (const std::exception& error) {
        break_capture();
        note_replay_fault(error.what());
    } catch (...) {
        break_capture();
        note_replay_fault("before_item exception");
    }
} // render thread, before each queue item
uint64_t s_logical_frame = 0;
void present_begin() { // gx2_render_present: the present's own commands are not the frame's
    ++s_logical_frame;
    if (gx2vk::capture::enabled()) gx2vk::capture::set_enabled(false);
}
uint64_t logical_frame() {
    return s_logical_frame;
}
bool replay_view(float& blend_t, bool& has_fix, float rot[9], float pivot[3]) {
    if (!s_replaying) return false;
    blend_t = s_blend_t;
    has_fix = s_fix.on;
    if (s_fix.on) {
        std::copy(s_fix.rot, s_fix.rot + 9, rot);
        std::copy(s_fix.pivot, s_fix.pivot + 3, pivot);
    }
    return true;
}

void keep(Gx2AsyncWork* work) {
    if (s_pending_replay) {
        delete work;
        return;
    }
    try {
        sync_capture();
        if (!capturing() || !work->replayable()) {
            delete work;
            return;
        }
        // A loading screen without presents must not grow without bound.
        if (s_works.size() >= 400000) {
            delete work;
            break_capture();
            return;
        }
        s_works.push_back(work);
    } catch (const std::exception& error) {
        delete work;
        break_capture();
        note_replay_fault(error.what());
    } catch (...) {
        delete work;
        break_capture();
        note_replay_fault("keep exception");
    }
}

void draw_unit(uint64_t key, bool transparent) {
    const uint64_t full = mix(key, ++s_occurrences[key]);
    if (s_replaying) {
        s_replay_cur = nullptr;
        s_replay_prev = nullptr;
        // By key, not position: a draw skipped in one run (shader still
        // compiling) must not shift every later draw onto other data.
        if (const auto it = s_cur->by_key.find(full); it != s_cur->by_key.end()) {
            s_replay_cur = &s_cur->units[it->second];
            if (!no_blend())
                if (const auto prev = s_last->occurrences.find(key); prev != s_last->occurrences.end()) {
                    // One previous draw: that one. Several (NPCs, ink drops
                    // with one mesh, whose order changes with sorting): the
                    // most similar. Very many (particles): no blending.
                    const uint32_t count = prev->second;
                    auto unit_of = [&](uint32_t occurrence) -> const Unit* {
                        const auto it = s_last->by_key.find(mix(key, occurrence));
                        return it != s_last->by_key.end() ? &s_last->units[it->second] : nullptr;
                    };
                    if (count == 1)
                        s_replay_prev = unit_of(1);
                    else if (count <= 48) {
                        double best = 1e300;
                        for (uint32_t occurrence = 1; occurrence <= count && best > 0; ++occurrence)
                            if (const Unit* candidate = unit_of(occurrence)) {
                                const double d = distance(*s_cur, *s_replay_cur, *s_last, *candidate);
                                if (d < best) {
                                    best = d;
                                    s_replay_prev = candidate;
                                }
                            }
                    }
                }
        } else {
            ++s_stats.missing;
        }
        ++s_stats.units;
        if (s_replay_prev) ++s_stats.matched;
        return;
    }
    s_unit = UINT32_MAX;
    sync_capture();
    if (!capturing()) return;
    if (s_cur->units.size() >= kMaxRecordedUnits) {
        break_capture();
        return;
    }
    try {
        const uint32_t idx = uint32_t(s_cur->units.size());
        s_cur->units.push_back({});
        s_cur->units.back().key = full;
        s_cur->units.back().base = key;
        s_cur->units.back().transparent = transparent;
        s_cur->by_key.emplace(full, idx);
        s_cur->occurrences[key] = s_occurrences[key];
        s_unit = idx;
        gx2vk::capture::set_unit(s_unit);
    } catch (const std::exception& error) {
        break_capture();
        note_replay_fault(error.what());
    } catch (...) {
        break_capture();
        note_replay_fault("draw_unit exception");
    }
}

void record_uf(unsigned stage, const void* data, size_t size) {
    if (s_unit == UINT32_MAX || s_unit >= s_cur->units.size() || !capturing() || stage > 1) return;
    if (size > kMaxRecordedBytes || s_cur->bytes.size() > kMaxRecordedBytes - size) {
        break_capture();
        return;
    }
    try {
        const uint32_t offset = s_cur->put(data, size);
        Unit& unit = s_cur->units[s_unit];
        unit.uf_off[stage] = offset;
        unit.uf_size[stage] = uint32_t(size);
        unit.has_uf[stage] = true;
    } catch (const std::exception& error) {
        break_capture();
        note_replay_fault(error.what());
    } catch (...) {
        break_capture();
        note_replay_fault("record_uf exception");
    }
}
void record_bank(unsigned stage, unsigned bank, const void* data, size_t size) {
    if (s_unit == UINT32_MAX || s_unit >= s_cur->units.size() || !capturing() || stage > 1 || bank > 15) return;
    if (size > kMaxRecordedBytes || s_cur->bytes.size() > kMaxRecordedBytes - size) {
        break_capture();
        return;
    }
    try {
        const uint32_t offset = s_cur->put(data, size);
        Unit& unit = s_cur->units[s_unit];
        unit.bank_off[stage][bank] = offset;
        unit.bank_size[stage][bank] = uint32_t(size);
        unit.bank_mask[stage] |= 1u << bank;
    } catch (const std::exception& error) {
        break_capture();
        note_replay_fault(error.what());
    } catch (...) {
        break_capture();
        note_replay_fault("record_bank exception");
    }
}

bool replay_uf(unsigned stage, void* data, size_t size) {
    if (!s_replaying || !s_replay_cur || stage > 1 || !s_replay_cur->has_uf[stage] ||
        s_replay_cur->uf_size[stage] != size || size_t(s_replay_cur->uf_off[stage]) + size > s_cur->bytes.size())
        return false;
    const uint8_t* cur = s_cur->bytes.data() + s_replay_cur->uf_off[stage];
    if (s_replay_prev && s_replay_prev->has_uf[stage] && s_replay_prev->uf_size[stage] == size &&
        size_t(s_replay_prev->uf_off[stage]) + size <= s_last->bytes.size())
        blend_bytes(s_last->bytes.data() + s_replay_prev->uf_off[stage], cur, static_cast<uint8_t*>(data), size, true);
    else
        std::memcpy(data, cur, size);
    return true;
}
const uint8_t* replay_bank(unsigned stage, unsigned bank, size_t size) {
    if (!s_replaying || !s_replay_cur || stage > 1 || bank > 15 || !(s_replay_cur->bank_mask[stage] >> bank & 1) ||
        s_replay_cur->bank_size[stage][bank] != size ||
        size_t(s_replay_cur->bank_off[stage][bank]) + size > s_cur->bytes.size())
        return nullptr;
    const uint8_t* cur = s_cur->bytes.data() + s_replay_cur->bank_off[stage][bank];
    s_blend.resize(size);
    if (s_replay_prev && (s_replay_prev->bank_mask[stage] >> bank & 1) && s_replay_prev->bank_size[stage][bank] == size &&
        size_t(s_replay_prev->bank_off[stage][bank]) + size <= s_last->bytes.size())
        blend_bytes(s_last->bytes.data() + s_replay_prev->bank_off[stage][bank], cur, s_blend.data(), size, false);
    else
        std::memcpy(s_blend.data(), cur, size);
    return s_blend.data();
}

namespace {
void rotate() {
    std::swap(s_last, s_cur); // the finished frame becomes N-1
    s_cur->clear();
    s_occurrences.clear();
    s_unit = UINT32_MAX;
    s_frame_broken = false;
    s_prev_cache.clear();
    s_span_owner.clear();
    // The next frame is captured from its first command (before_item), not
    // from here: the present that follows is not part of it.
    s_capture_armed = true;
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}
bool env_off(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '0';
}

// The rate to present at: the menu's choice, lowered to what the display can
// show. With G-Sync/FreeSync (IMMEDIATE presents) the display follows each
// picture as long as the rate stays inside its range, so the limit is a
// little below the refresh rate; above it pictures would tear or be dropped.
// Without it (MAILBOX) the display shows one picture per refresh at most, so
// the limit is the refresh rate itself and the presents are locked to the
// vertical blank. RBREW_INTERP_UNCAPPED=1 (measurements): no limit.
struct Target {
    double rate = 1.0, fps = 60.0, hz = 0.0; // rate: presents per game step
    bool capped = false, vrr = false, lock = false;
};
constexpr double kStepsPerSecond = 60000.0 / 1001.0;
Target compute_target(int frame_rate, bool vrr, bool fixed_refresh, double steps_per_second = kStepsPerSecond) {
    Target t;
    t.hz = gx2vk::display_refresh_hz();
    if (!std::isfinite(t.hz) || t.hz < 0.0) t.hz = 0.0;
    t.vrr = vrr;
    const bool display = frame_rate == rbrew_settings::kFrameRateDisplay;
    // 120, 165, 180 and 240 are 2, 2.75, 3 and 4 presents per 60 Hz step (the
    // game steps 59.94 times a second, so 120 is 119.88 pictures a second).
    // In 30 Hz scenes (e.g. plaza_60fps=0), steps_per_second is ~29.97 so rate
    // scales to hit the full display/chosen refresh rate instead of half of it.
    double fps = display ? (t.hz > 0.0 ? t.hz : 120.0) : double(frame_rate) * kStepsPerSecond / 60.0;
    if (!std::isfinite(fps) || fps < 60.0) fps = 60.0;
    static const bool uncapped = env_on("RBREW_INTERP_UNCAPPED");
    if (t.hz > 0.0 && !uncapped) {
        if (vrr) {
            const double limit = t.hz - std::max(3.0, t.hz * 0.02);
            if (fps > limit) {
                t.capped = !display;
                fps = limit;
            }
        } else if (fixed_refresh && fps > t.hz * 0.995) {
            t.capped = !display && fps > t.hz * 1.005;
            fps = t.hz;
            t.lock = true;
        }
    }
    t.fps = fps;
    t.rate = std::clamp(fps / steps_per_second, 1.0, double(kMaxPerStep - 2));
    if (t.rate < 1.02) t.rate = 1.0;
    return t;
}
Target current_target(double steps_per_second = kStepsPerSecond) {
    const VkPresentModeKHR mode = gx2vk::present_mode();
    const bool immediate = mode == VK_PRESENT_MODE_IMMEDIATE_KHR;
    const bool gsync = rbrew_settings::g_gsync.load(std::memory_order_relaxed);
    return compute_target(rbrew_settings::g_frame_rate.load(std::memory_order_relaxed), gsync && immediate, !immediate,
                          steps_per_second);
}

// A rate of a/b presents per step (b <= 8) puts every picture on a multiple
// of 1/a between two game frames: frame N itself is one of them (once every b
// steps), and no picture is ever closer to N-1 than 1/a, so the whole
// schedule runs 1/a of a step earlier. Other rates (the display's refresh
// rate) have no such grid: lo = 0, and every picture is an in-between frame.
struct Grid {
    double lo = 0.0;   // smallest position a picture can have
    double cost = 0.0; // in-between frames drawn per game step
    unsigned a = 0;    // 0: no grid
};
Grid grid_of(double rate) {
    for (unsigned b = 1; b <= 8; ++b) {
        const double a = rate * b, whole = std::round(a);
        if (whole >= 1.0 && std::fabs(a - whole) < 1e-4) return {1.0 / whole, rate - 1.0 / b, unsigned(whole)};
    }
    return {0.0, rate, 0};
}
// The controller's levels, highest first: evenly spaced rates whose cost
// (in-between frames drawn per game step) is lower than every rate above.
// A rate a/b shows frame N once every b steps, so it costs (a - 1) / b: the
// whole rates are the cheap ones (2: 1 per step, 3: 2), and 2.5 costs as much
// as 3 (dropped when 3 fits under the target). 165 with G-Sync (cap 161.7 at
// 163.97 Hz: 2.698 per step) gives 8/3 (159.8 FPS, 2.33), 5/2 (2), 2 (1), 1.
// A non-grid target (2.698: every picture an in-between frame) is kept only
// when no grid rate is within 3% below it, or when the presents are locked to
// a fixed refresh (which needs that exact rate).
struct Level {
    double rate, cost;
};
std::vector<Level> ladder(double target, bool lock) {
    std::vector<Level> all;
    const Grid top = grid_of(target);
    all.push_back({target, top.cost});
    for (unsigned b = 1; b <= 4; ++b)
        for (unsigned a = b; double(a) <= target * b + 1e-6; ++a) all.push_back({double(a) / b, double(a - 1) / b});
    std::sort(all.begin(), all.end(), [](const Level& x, const Level& y) {
        return x.rate != y.rate ? x.rate > y.rate : x.cost < y.cost;
    });
    std::vector<Level> out;
    for (const Level& l : all)
        if (out.empty() || (l.cost < out.back().cost - 1e-6 && l.rate < out.back().rate - 1e-6)) out.push_back(l);
    if (!lock && !top.a && out.size() > 1 && out[1].rate >= target * 0.97) out.erase(out.begin());
    return out;
}

unsigned take_slot() {
    const unsigned slot = s_next_slot;
    s_next_slot = (s_next_slot + 1) % kSlots;
    return slot;
}
uint64_t effective_due(const Shot& shot) {
    // Never two pictures closer than 65% of a present interval: late ones
    // spread out smoothly instead of bunching into alternating short/long gaps.
    const uint64_t gap = uint64_t(0.65 * s_interval_ns / std::max(1.0, s_rate));
    return std::max(shot.due, s_last_present_ns + gap);
}
// Presents the queued pictures whose time has come, in order. Before each
// replay of a burst a picture due before that replay would end is waited for
// (the next chance is a whole replay later: it went out that much late, then
// the next one early, once per step: 8.9 / 3.5 ms instead of 6.2 / 6.2 at
// 162 FPS); the waits of one burst are bounded.
void present_due(bool in_burst, int site) {
    // Keep burst wait bounded (<= 1.2 ms): catches pictures that would otherwise
    // miss their due time by ~0.5-1 ms during a replay without stalling the game.
    const uint64_t burst_wait = uint64_t(std::clamp(s_replay_avg_ns * 0.85 + 100000.0, 150000.0, 1200000.0));
    constexpr uint64_t kBurstWaitBudgetNs = 1200000;
    while (!s_shots.empty()) {
        const Shot shot = s_shots.front();
        const uint64_t due = effective_due(shot);
        uint64_t now = now_ns();
        if (now < due) {
            if (!in_burst || due - now > burst_wait || s_burst_waited_ns + (due - now) > kBurstWaitBudgetNs) break;
            s_burst_waited_ns += due - now;
            while ((now = now_ns()) < due) _mm_pause();
        }
        s_shots.erase(s_shots.begin());
        gx2_render_interp_present(shot.slot);
        s_last_present_ns = now_ns();
        const double late_ms = double(now - due) / 1e6;
        // RBREW_INTERP_PRESENT_LOG=<file>: per picture: present time, its due
        // time (us), position t, where it went out (0 render loop, 1 between
        // two draws, 2 before a replay), how long the present call took (us).
        static FILE* log = [] {
            const char* p = std::getenv("RBREW_INTERP_PRESENT_LOG");
            return p && *p ? std::fopen(p, "w") : nullptr;
        }();
        if (log) {
            std::fprintf(log, "%llu %llu %.3f %d %llu\n", (unsigned long long)(now / 1000),
                         (unsigned long long)(due / 1000),
                         shot.t, site, (unsigned long long)((s_last_present_ns - now) / 1000));
            if ((s_stats.presents & 63) == 0) std::fflush(log);
        }
        s_stats.late_ms += late_ms;
        s_stats.late_max_ms = std::max(s_stats.late_max_ms, late_ms);
        if (late_ms > 1.0) ++s_stats.late1;
        ++s_stats.presents;
        if (shot.t < 1.0f)
            ++s_stats.between;
        else
            ++s_stats.held;
    }
}
void drop_shots() {
    s_stats.dropped += unsigned(s_shots.size());
    s_shots.clear();
}
} // namespace

Status status() {
    return s_status;
}
Status describe(int frame_rate, bool gsync) {
    const Target t = compute_target(frame_rate, gsync, !gsync);
    Status s;
    s.target_fps = float(t.fps);
    s.display_hz = float(t.hz);
    s.capped = t.capped;
    s.vrr = t.vrr;
    s.locked = t.lock;
    return s;
}

void set_view(const rbrew_aim::ViewTag& view) {
    s_incoming_view = view;
}

bool frame_complete(unsigned scan_index) {
    const uint64_t now = now_ns();
    s_cur->view = s_held_view;
    s_held_view = s_incoming_view;
    s_incoming_view = {};
    if (s_last_complete) {
        const double interval = double(now - s_last_complete);
        if (interval > 4e6 && interval < 50e6) s_interval_ns = s_interval_ns * 0.9 + interval * 0.1;
    }
    s_last_complete_interval_ns = s_last_complete ? double(now - s_last_complete) : 0;
    s_last_complete = now;
    ++s_stats.frames;
    s_stats.busy_ms += double(s_frame_busy_ns) / 1e6;
    s_busy_avg_ns = s_busy_avg_ns ? s_busy_avg_ns * 0.8 + double(s_frame_busy_ns) * 0.2 : double(s_frame_busy_ns);
    s_busy_last_ns = s_frame_busy_ns;
    s_frame_busy_ns = 0;
    report_stats();
    gx2vk::capture::set_enabled(false);
    gx2vk::capture::finish_frame(); // the captured commands become the replay frame
    s_vk_replay = vk_mode() && !s_capture_mid_frame && gx2vk::capture::replay_complete();
    bool replay = capturing() && !s_frame_broken && now >= s_fault_until_ns && !s_last->units.empty() &&
                  !s_cur->units.empty() &&
                  (s_vk_replay || !s_works.empty());
    // Why a frame has no in-between frames (stats "why"): not captured,
    // capture broken, no recorded draws, Vulkan capture incomplete, capture
    // started mid-frame.
    if (s_frame_broken)
        ++s_stats.why_broken;
    else if (!capturing())
        ++s_stats.why_capture;
    else if (s_last->units.empty() || s_cur->units.empty())
        ++s_stats.why_units;
    if (vk_mode() && !gx2vk::capture::replay_complete()) ++s_stats.why_incomplete;
    if (vk_mode() && s_capture_mid_frame) ++s_stats.why_mid;
    // RBREW_INTERP_ALWAYS=1 (diagnostics): the target rate, whatever the game's speed.
    static const bool always = env_on("RBREW_INTERP_ALWAYS");
    // A single hitch (streaming, 30-70 ms) counts as at most 1.5 frames: at
    // full weight it kept the average above the limit for 20+ frames and cut
    // the in-between frames for half a second (2026-10-27).
    // RBREW_INTERP_HITCH_CLAMP=0: full weight.
    // Detect 30 Hz game cadence (e.g. plaza_60fps=0: 33.37 ms per drawn step)
    // so the controller budget and target rate match the actual step period.
    const bool half_rate = s_interval_ns > kLogicFrameNs * 1.7 && s_interval_ns < kLogicFrameNs * 2.3;
    const double expected_step_ns = half_rate ? 2.0 * kLogicFrameNs : kLogicFrameNs;
    const double steps_per_second = 1e9 / expected_step_ns;
    static const bool clamp_hitch = !env_off("RBREW_INTERP_HITCH_CLAMP");
    if (s_last_complete_interval_ns > 0 && s_last_complete_interval_ns < expected_step_ns * 6.0) {
        const double sample =
            clamp_hitch ? std::min(s_last_complete_interval_ns, expected_step_ns * 1.5) : s_last_complete_interval_ns;
        s_game_interval_ns = s_game_interval_ns ? s_game_interval_ns * 0.85 + sample * 0.15 : sample;
    }
    const Target target = current_target(steps_per_second);
    static std::vector<Level> s_ladder;
    static double s_ladder_target = -1.0;
    static bool s_ladder_lock = false;
    if (target.rate != s_ladder_target || target.lock != s_ladder_lock) {
        s_ladder = ladder(target.rate, target.lock);
        if (s_ladder.empty()) s_ladder.push_back({1.0, 0.0});
        s_ladder_target = target.rate;
        s_ladder_lock = target.lock;
        s_level = 0;
    }
    const unsigned levels = unsigned(s_ladder.size() - 1);
    static double s_busy_hist[64]{};
    static unsigned s_busy_n = 0;
    if (s_last_complete_interval_ns > 0 && s_last_complete_interval_ns < expected_step_ns * 6.0)
        s_busy_hist[s_busy_n++ % 64] = double(s_busy_last_ns);
    double busy_p80 = 0;
    {
        const unsigned n = std::min(s_busy_n, 64u);
        if (n) {
            double sorted[64];
            std::copy(s_busy_hist, s_busy_hist + n, sorted);
            std::nth_element(sorted, sorted + (n * 4) / 5, sorted + n);
            busy_p80 = sorted[std::min(n - 1, (n * 4) / 5)];
        }
    }
    // Detect loading screens and hitches before running the controller so a
    // loading transition never drops the rate or inflates s_raise_wait.
    static unsigned s_after_hitch = 0;
    if (s_last_complete_interval_ns > expected_step_ns * 3)
        s_after_hitch = 10; // > 50 ms at 60 Hz, > 100 ms at 30 Hz
    else if (s_after_hitch)
        --s_after_hitch;
    static const bool load_guard = !env_off("RBREW_INTERP_LOAD_GUARD");
    static unsigned s_after_load = 0;
    if (rbrew_benchmark::frame_draws.load(std::memory_order_relaxed) < 300)
        s_after_load = 60;
    else if (s_after_load)
        --s_after_load;
    const bool in_transition = (s_after_hitch && clamp_hitch) || (s_after_load && load_guard);

    const size_t frame_commands = gx2vk::capture::replay_command_count();
    const double replay_cost = s_replay_ns_per_cmd > 0 && frame_commands
                                   ? s_replay_ns_per_cmd * double(frame_commands)
                                   : (s_replay_avg_ns > 0 ? s_replay_avg_ns : 1.5e6);
    auto load = [&](unsigned level) { return busy_p80 + s_ladder[level].cost * replay_cost; };
    static unsigned s_over_frames = 0, s_slow_frames = 0, s_fit_frames = 0, s_since_raise = 100000,
                    s_raise_wait = 45;
    if (s_level > levels) s_level = levels;
    const double step_budget = expected_step_ns;
    if (in_transition) {
        s_over_frames = s_slow_frames = 0;
        if (s_after_load == 60) {
            // Fresh scene load: start ready to climb cleanly instead of staying
            // trapped at 60 FPS for 20 s after a heavy plaza frame.
            s_raise_wait = 45;
            s_fit_frames = 0;
        }
    } else {
        s_over_frames = s_level < levels && load(s_level) > 0.94 * step_budget ? s_over_frames + 1 : 0;
        s_slow_frames =
            s_level < levels && s_game_interval_ns > expected_step_ns * 1.05 && load(s_level) > 0.78 * step_budget
                ? s_slow_frames + 1
                : 0;
        s_fit_frames =
            s_level > 0 && load(s_level - 1) < 0.84 * step_budget && s_game_interval_ns < expected_step_ns * 1.04
                ? s_fit_frames + 1
                : 0;
        if (s_since_raise < 100000 && ++s_since_raise == 240) s_raise_wait = 45; // held for 4 s
        if (s_over_frames > 10 || s_slow_frames > 24) {
            unsigned to = s_level + 1;
            if (s_over_frames > 10)
                while (to < levels && load(to) > 0.88 * step_budget) ++to;
            s_level = std::min(to, levels);
            ++s_stats.level_downs;
            s_over_frames = s_slow_frames = s_fit_frames = 0;
            if (s_since_raise < 240) s_raise_wait = std::min(s_raise_wait * 2, 180u);
            s_since_raise = 100000;
        } else if (s_fit_frames >= s_raise_wait) {
            if (load(0) < 0.70 * step_budget)
                s_level = 0; // light scene (shooting range / menus): jump straight to target
            else
                --s_level;
            s_fit_frames = 0;
            s_since_raise = 0;
        }
    }
    if (always) s_level = 0;
    double rate = s_ladder[s_level].rate;
    // RBREW_INTERP_RATE=<presents per step> (testing): this rate, no controller.
    static const double forced_rate = [] {
        const char* v = std::getenv("RBREW_INTERP_RATE");
        return v && std::atof(v) >= 1.0 ? std::min(std::atof(v), double(kMaxPerStep - 2)) : 0.0;
    }();
    if (forced_rate > 0.0) rate = forced_rate;
    // The GX2-level replay (RBREW_INTERP_VK=0) draws one in-between frame.
    if (!s_vk_replay && rate > 1.0) rate = 2.0;
    if (s_after_hitch && clamp_hitch && replay) {
        replay = false;
        ++s_stats.skipped;
    }
    if (s_after_load && load_guard && replay) {
        replay = false;
        ++s_stats.skipped;
    }
    s_status.target_fps = float(target.fps);
    s_status.display_hz = float(target.hz);
    s_status.capped = target.capped;
    s_status.vrr = target.vrr;
    s_status.locked = target.lock;
    s_status.level_fps = float(rate * steps_per_second);
    s_status.active = replay && rate > 1.0;
    if (!replay || rate <= 1.0) {
        // This frame is presented as it is, now: pictures still waiting are
        // older than it and are given up.
        gx2vk::set_capture_hold(0);
        drop_shots();
        s_schedule_reset = true;
        s_rate = 1.0;
        ++s_stats.direct;
        drop_works(s_works);
        if (capturing() || (s_frame_broken && !s_replaying && s_wanted.load(std::memory_order_relaxed)))
            rotate();
        else {
            s_cur->clear();
            s_last->clear();
            s_occurrences.clear();
            s_unit = UINT32_MAX;
            s_frame_broken = false;
        }
        return false;
    }
    // The plan of this step: one picture per present interval, each at the
    // position between N-1 and N where its present time falls. `s_phase` is
    // the position of the next picture and runs on across steps, so any rate
    // works, whole or not (165: 2.75 per step, positions on elevenths).
    const double step_ns =
        std::fabs(s_interval_ns - expected_step_ns) < 0.03 * expected_step_ns ? expected_step_ns : s_interval_ns;
    const Grid grid = grid_of(rate);
    const double advance = 1.0 / rate;
    if (s_schedule_reset) {
        s_schedule_reset = false;
        s_anchor_ns = double(now);
        s_jitter_ns = 0;
        s_phase = grid.a ? grid.lo : advance;
        s_phase_lo = grid.lo;
        s_lock_offset_ns = 0;
    } else {
        // The schedule follows a smoothed clock of the game's frames, not
        // each frame's own arrival: presents stay evenly spaced when a frame
        // comes a little early or late.
        const double predicted = s_anchor_ns + step_ns, error = double(now) - predicted;
        if (std::fabs(error) > step_ns * 0.5)
            s_anchor_ns = double(now);
        else {
            s_anchor_ns = predicted + 0.08 * error;
            s_jitter_ns = s_jitter_ns * 0.98 + std::fabs(error) * 0.02;
        }
        // A changed rate: back onto its grid; the schedule's offset follows gently.
        if (grid.a) s_phase = std::max(1.0, std::round(s_phase * grid.a)) / grid.a;
        s_phase_lo += std::clamp(grid.lo - s_phase_lo, -0.02, 0.02);
    }
    s_rate = rate;
    // RBREW_INTERP_LEAD_MS=<ms> (testing): fixed time from a frame's completion
    // to its first picture, instead of the measured replay time.
    static const double forced_lead_ns = [] {
        const char* v = std::getenv("RBREW_INTERP_LEAD_MS");
        return v && *v ? std::atof(v) * 1e6 : -1.0;
    }();
    const double lead_ns = forced_lead_ns >= 0 ? forced_lead_ns : s_lead_ns + std::min(2.0 * s_jitter_ns, 2e6);
    s_plan.count = 0;
    if (s_phase < 0.0 || s_phase > 1.0 + 1e-6) s_phase = grid.a ? grid.lo : advance;
    while (s_phase <= 1.0 + 1e-6) {
        if (s_plan.count < kMaxPerStep) {
            const double t = s_phase >= 1.0 - 1e-6 ? 1.0 : s_phase;
            s_plan.t[s_plan.count] = float(t);
            s_plan.due[s_plan.count] = uint64_t(s_anchor_ns + lead_ns + std::max(0.0, t - s_phase_lo) * step_ns);
            ++s_plan.count;
        }
        s_phase += advance;
    }
    s_phase = std::max(0.0, s_phase - 1.0);
    if (s_plan.count == 0) {
        gx2vk::set_capture_hold(0);
        drop_shots();
        s_schedule_reset = true;
        s_rate = 1.0;
        ++s_stats.direct;
        drop_works(s_works);
        rotate();
        return false;
    }
    // Fixed refresh (no G-Sync), one picture per refresh: the presents are
    // pulled gently to the middle between two vertical blanks, so every refresh
    // gets exactly one new picture (MAILBOX shows the newest at the blank).
    // Keep s_lock_offset_ns in [0, p) without discontinuous +-p wraps so
    // due[0] never jumps backward before replay finishes or skips a refresh.
    uint64_t period = 0, vblank = 0;
    if (target.lock && s_level == 0 && forced_rate <= 0.0 && s_plan.count && gx2vk::display_vblank(period, vblank)) {
        const double p = double(period);
        double x = std::fmod(double(s_plan.due[0]) + s_lock_offset_ns - (double(vblank) + 0.5 * p), p);
        if (x < 0) x += p;
        if (x > 0.5 * p) x -= p; // how far after (before) the middle the first picture would go out
        const double step = std::clamp(-0.10 * x, -150000.0, 150000.0);
        s_lock_offset_ns = std::clamp(s_lock_offset_ns + step, 0.0, p);
        s_stats.lock_error_ms += std::fabs(x) / 1e6;
        ++s_stats.lock_samples;
    } else
        s_lock_offset_ns *= 0.95;
    for (unsigned i = 0; i < s_plan.count; ++i)
        s_plan.due[i] = uint64_t(std::max(0.0, double(s_plan.due[i]) + s_lock_offset_ns));
    // Room for this step's pictures among the copies (a queue this long means
    // the presents are far behind: the oldest are given up).
    while (!s_shots.empty() && s_shots.size() + s_plan.count + 1 > kSlots - 1) {
        s_shots.erase(s_shots.begin());
        ++s_stats.dropped;
    }
    // Frame N itself: one of the pictures on a grid (its position 1), and
    // what is shown when the in-between frames are discarded (camera cut).
    s_plan.held_slot = take_slot();
    gx2_render_interp_keep(scan_index, s_plan.held_slot, 0);
    s_scan_index = scan_index;
    s_replay_works.swap(s_works);
    drop_works(s_works);
    s_pending_replay = true;
    return true;
}

namespace {
// The in-between frames of the step, drawn one after another right after
// frame N, before the next frame's commands touch the renderer: each is
// copied aside and waits for its present time.
void run_replays() {
    const uint64_t start = now_ns();
    s_replaying = true;
    s_occurrences.clear();
    s_burst_waited_ns = 0;
    bool ok = true;
    bool vk_replayed = false;
    unsigned made = 0;
    const size_t queued_before = s_shots.size();
    const unsigned presents_before = s_stats.presents;
    if (s_vk_replay) {
        s_stats.set_binds += unsigned(gx2vk::capture::replay_set_binds());
        // A camera cut or a very fast turn: many matrices jump and keep
        // frame N while others of the same camera blend; the in-between
        // frame then shows parts of two views (a whole quarter of the
        // screen in Octo Valley). The step's in-between frames are
        // discarded; frame N is shown in their place.
        // RBREW_INTERP_CUT_GUARD=0: keep them.
        static const bool cut_guard = !env_off("RBREW_INTERP_CUT_GUARD");
        auto cut = [] {
            const unsigned all = s_blended_matrices + s_jumped_matrices;
            return all >= 20 && s_jumped_matrices * 4 > all;
        };
        s_blended_matrices = s_jumped_matrices = 0;
        bool replayed_n = false;
        try {
        for (unsigned i = 0; i < s_plan.count; ++i) {
            s_blend_t = s_plan.t[i];
            prepare_view_fix(s_blend_t);
            // Reuse raw frame N only when mouse did not move between Swap N-1
            // and Swap N; otherwise replay t=1.0 rotated to Swap N so the
            // camera never snaps backwards by 16.7 ms on exact frames.
            if (s_plan.t[i] >= 1.0f && !s_fix.on) continue;
            // Before each replay (the first too: the previous step's last
            // picture is often due right now): pictures whose time has come,
            // or comes during the replay, go out.
            s_replaying = false;
            present_due(true, 2);
            s_replaying = true;
            const uint64_t replay_start = now_ns();
            // N's upload data must outlive every in-between frame's GPU work;
            // presents between them close submissions.
            gx2vk::retain_until_current_submission(s_frame_first_serial, 2 * (s_plan.count - i) + 6);
            if (s_fix.on) {
                ++s_stats.look;
                s_stats.look_yaw += std::fabs(s_fix.yaw);
            }
            s_substituted.clear();
            s_by_content.clear();
            s_occurrences.clear();
            vk_replayed = true;
            gx2vk::capture::replay(&substitute);
            if (cut_guard && cut()) {
                ok = false;
                ++s_stats.cuts;
                break;
            }
            s_replaying = false;
            const unsigned slot = take_slot();
            gx2_render_interp_keep(s_scan_index, slot, made + 1);
            s_replaying = true;
            s_shots.push_back({slot, s_plan.due[i], s_plan.t[i]});
            if (s_plan.t[i] >= 1.0f) replayed_n = true;
            ++made;
            const uint64_t done = now_ns();
            const double took = double(done - replay_start);
            s_replay_avg_ns = s_replay_avg_ns ? s_replay_avg_ns * 0.9 + took * 0.1 : took;
            if (const size_t commands = gx2vk::capture::replay_command_count()) {
                const double per = took / double(commands);
                s_replay_ns_per_cmd = s_replay_ns_per_cmd ? s_replay_ns_per_cmd * 0.9 + per * 0.1 : per;
            }
            s_stats.replay_ms += took / 1e6;
            ++s_stats.replays;
            if (made == 1) {
                // From N's completion to its first picture: smooth asymmetrically
                // (fast rise to prevent late presents, slow decay to prevent 60 Hz cadence beat).
                const double lead = double(done - s_last_complete) * 1.1 + 150000.0;
                const double alpha = lead > s_lead_ns ? 0.15 : 0.02;
                s_lead_ns = std::clamp(s_lead_ns * (1.0 - alpha) + lead * alpha, 0.3e6, 8e6);
            }
        }
        } catch (const std::exception& error) {
            ok = false;
            ++s_stats.failed;
            note_replay_fault(error.what());
        } catch (...) {
            ok = false;
            ++s_stats.failed;
            note_replay_fault("unknown exception");
        }
        s_blend_t = 0.5f;
        s_fix.on = false;
        s_fix.cur_on = false;
        if (!ok) replayed_n = false;
        if (ok && s_plan.count && s_plan.t[s_plan.count - 1] >= 1.0f && !replayed_n)
            s_shots.push_back({s_plan.held_slot, s_plan.due[s_plan.count - 1], 1.0f});
    } else {
        try {
            // Publish finished occlusion queries meanwhile: the game reads the
            // previous frame's results early in its frame (gx2_draw.cpp).
            s_blend_t = s_plan.t[0] < 1.0f ? s_plan.t[0] : 0.5f;
            unsigned n = 0;
            for (Gx2AsyncWork* work : s_replay_works) {
                work->run();
                if ((++n & 31) == 0) gx2_render_query_poll_completed();
            }
            s_replaying = false;
            const unsigned slot = take_slot();
            gx2_render_interp_keep(s_scan_index, slot, 1);
            s_shots.push_back({slot, s_plan.due[0], s_plan.t[0]});
            made = 1;
            const double took = double(now_ns() - start);
            s_replay_avg_ns = s_replay_avg_ns ? s_replay_avg_ns * 0.9 + took * 0.1 : took;
            s_stats.replay_ms += took / 1e6;
            ++s_stats.replays;
            s_lead_ns = std::clamp(s_lead_ns * 0.9 + (took * 1.1 + 150000.0) * 0.1, 0.3e6, 8e6);
        } catch (const std::exception& error) {
            ok = false;
            ++s_stats.failed;
            note_replay_fault(error.what());
        } catch (...) {
            ok = false;
            ++s_stats.failed;
            note_replay_fault("unknown exception");
        }
        s_blend_t = 0.5f;
    }
    s_replaying = false;
    s_replay_cur = s_replay_prev = nullptr;
    drop_works(s_replay_works);
    // Frame N's host-visible buffers stay untouched until the GPU ran the
    // last replay (replay_may_read): it is in the submission recorded now.
    // Guard on vk_replayed (not made) so a camera cut on the first replay
    // still protects frame N's buffers and resets command tracking after
    // replay() mutated the command buffer's bound state.
    if (s_vk_replay && vk_replayed) {
        s_holds[s_hold_next] = {s_frame_first_serial, gx2vk::recording_serial() + 2};
        s_hold_next = (s_hold_next + 1) % kHolds;
        gx2_render_reset_command_tracking();
    }
    gx2vk::set_capture_hold(0);
    const bool has_n = s_plan.count && s_plan.t[s_plan.count - 1] >= 1.0f;
    if (!ok) {
        // The step's in-between pictures that did not go out yet are given
        // up; frame N takes the first one's time. (Pictures presented during
        // the burst came off the front of the queue: older ones first.)
        const size_t presented = s_stats.presents >= presents_before ? size_t(s_stats.presents - presents_before) : 0;
        const size_t gone = std::min<size_t>(queued_before, presented);
        const size_t keep = std::min(s_shots.size(), queued_before - gone);
        s_stats.dropped += unsigned(s_shots.size() - keep);
        s_shots.resize(keep);
        const uint64_t fallback_due = s_plan.count ? s_plan.due[0] : now_ns();
        s_shots.push_back({s_plan.held_slot, fallback_due, 1.0f});
    } else if (!s_vk_replay && has_n)
        s_shots.push_back({s_plan.held_slot, s_plan.due[s_plan.count - 1], 1.0f});
    rotate();
}
} // namespace

bool replay_may_read(uint64_t last_serial) {
    // Also protect buffers touched during the frame currently being captured
    // or awaiting replay: a mid-frame poll() submit can complete last_serial
    // before run_replays() runs and pushes s_holds.
    if ((gx2vk::capture::enabled() || s_pending_replay || s_replaying) && last_serial >= s_frame_first_serial) {
        ++s_stats.hazards;
        return vb_guard();
    }
    const uint64_t done = gx2vk::known_completed_serial();
    for (const Hold& hold : s_holds)
        if (hold.until > done && last_serial >= hold.from && last_serial <= hold.until) {
            ++s_stats.hazards;
            return vb_guard();
        }
    return false;
}

namespace {
void note_replay_fault(const char* what) {
    s_fault_until_ns = now_ns() + 10'000'000'000ull; // 10 s of plain frames
    char message[160];
    std::snprintf(message, sizeof(message), "frame interpolation replay failed (%s): plain frames for 10 s", what);
    SplatoonDiagnostics::RecordEvent(SplatoonDiagnostics::EventCategory::Engine,
                                     SplatoonDiagnostics::EventSeverity::Warning, 0, message);
}
void recover_service_fault(const char* what) {
    s_replaying = false;
    s_fix.on = false;
    s_fix.cur_on = false;
    s_blend_t = 0.5f;
    s_replay_cur = s_replay_prev = nullptr;
    drop_works(s_replay_works);
    drop_shots();
    s_schedule_reset = true;
    s_rate = 1.0;
    gx2vk::capture::set_enabled(false);
    gx2vk::capture::discard();
    gx2vk::set_capture_hold(0);
    gx2_render_reset_command_tracking();
    rotate();
    note_replay_fault(what);
}
} // namespace

void service() {
    if (s_pending_replay) {
        s_pending_replay = false;
        try {
            run_replays();
        } catch (const std::exception& error) {
            recover_service_fault(error.what());
        } catch (...) {
            recover_service_fault("unknown exception");
        }
    }
    try {
        present_due(false, 0);
    } catch (const std::exception& error) {
        recover_service_fault(error.what());
    } catch (...) {
        recover_service_fault("present_due exception");
    }
}

void poll() {
    if (s_shots.empty() || s_replaying || s_pending_replay) return;
    if (now_ns() < effective_due(s_shots.front())) return;
    try {
        present_due(false, 1);
    } catch (const std::exception& error) {
        recover_service_fault(error.what());
    } catch (...) {
        recover_service_fault("poll exception");
    }
}

uint64_t deadline_ns() {
    if (s_pending_replay) return 1;
    return s_shots.empty() ? 0 : std::max<uint64_t>(1, effective_due(s_shots.front()));
}
} // namespace gx2interp
