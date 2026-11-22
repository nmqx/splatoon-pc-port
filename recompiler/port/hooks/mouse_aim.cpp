// mouse_aim.cpp - the mouse turns the game's player camera directly.
//
// The player camera (one object per player, `this` below) is updated once per
// game frame by 025DAAF8, which calls in turn
//   025D4A2C  input: right stick and GamePad motion -> the camera's angles
//   025D7CC0  pose: eye and look-at point from those angles (025D18E0)
// The state 025D4A2C leaves for the pose:
//   this+0x1B8  vec3  horizontal look direction (sin yaw, 0, cos yaw). The
//                     stick and the gyro only ever rotate it by a step
//                     (0204EFD8), so the yaw is whatever this vector says.
//   this+0x1E0  vec3  the same, smoothed while this+0x1B4 > 0 (else a copy);
//                     the pose reads this one.
//   this+0x1F4  f32   yaw in radians the stick added (used by one motion mode).
//   this+0xFC   f32   pitch, normalised: 0 = this+0x1FC (-7.5 degrees),
//                     +1 = this+0x1F8 (60, up), -1 = this+0x200 (-75, down),
//                     linear in between (025D2074/025D2158). The game moves it
//                     20% per frame towards a target it derives from the
//                     stick's angle (this+0xE0) or the pad's attitude (+0xF4).
//   this+0x23C  u32   non-zero while the game holds the camera (stick ignored).
// The motion path is why mouse-as-gyro never felt right: the yaw rate goes
// through a tremor filter (025D40BC: gain below 1 for small steps, the rest
// banked at this+0x294 and paid back later), the pitch is the pad's absolute
// attitude with a gain of about 3.3, and both lag.
//
// So the mouse does not go through either input. The manifest patches the one
// call of 025D4A2C (`[patches]` 0x025DB0D8 = bl 0x02D0F000) to a thunk bound
// here. The thunk runs the game's function unchanged, then, before the pose is
// built in the same frame:
//   yaw    rotates this+0x1B8 and this+0x1E0 by the mouse step (and adds it to
//          this+0x1F4): exact, no filter, unbounded;
//   pitch  keeps the game's own value G (stick, gyro, game events all still
//          move it) and adds a mouse offset in degrees: the camera gets
//          clamp(G + offset) and G is put back before the next update, so the
//          game's smoothing never pulls the mouse movement back.
// With the mouse idle the yaw is never written and the pitch only while an
// offset is left: stick and gyro move the camera as before. The mouse is
// dropped while the game holds the camera (+0x23C) or gives the player no
// control of it (+0xF8 == 0, e.g. the title screen over the plaza). Measured
// (logs\aim\analyze.py on RBREW_AIM_AUTOTEST runs, plaza): yaw and pitch equal
// the commanded angles to 0.001 degree in the frame the mouse moved, also in
// the rendered view; no drift after back-and-forth; limits never stick.
#include "mouse_aim.h"
#include "gx2_camera_hooks.h"
#include "../runtime/rebrewu_runtime.h"
#include "../runtime/game_revision.h"
#include "../runtime/native_control_defaults.h"
#include "../runtime/port_settings.h"
#include "../os/gx2/gx2_render.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <chrono>
#include <thread>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rbrew_aim {
namespace {
// From the manifest [globals]: v16 0x02D0F000 / 0x025D4A2C, v272 0x02DDF000 / 0x026621E8.
constexpr uint32_t kThunk = RBREW_GAME_MOUSE_AIM_THUNK;    // free slot after the last import stub
constexpr uint32_t kCameraInput = RBREW_GAME_CAMERA_INPUT; // the function the patched call used to reach
constexpr uint32_t kDir = 0x1B8, kDirSmooth = 0x1E0, kStickYaw = 0x1F4, kPitch = 0xFC;
constexpr uint32_t kPitchMax = 0x1F8, kPitchMid = 0x1FC, kPitchMin = 0x200, kLocked = 0x23C;
// The camera's input weight: 025D4A2C sets it to 0 while the player may not
// turn the camera (025A9F44: title screen over the plaza, special states; the
// game then also drops the stick) and ramps it back to 1 afterwards.
constexpr uint32_t kInputWeight = 0xF8;
constexpr uint32_t kObjectSize = 0x320;
constexpr float kPi = 3.14159265358979f;

float rf(const uint8_t* mem, uint32_t a) {
    return float(rbrew_read_f32(const_cast<uint8_t*>(mem), a));
}
void wf(uint8_t* mem, uint32_t a, float v) {
    rbrew_write_f32(mem, a, v);
}

std::mutex s_lock;
// VPADRead side (under s_lock).
float s_pending_yaw = 0, s_pending_pitch = 0; // degrees not yet given to the camera
bool s_stick_pitch = false;                   // the right stick is pitching the camera
uint64_t s_vpad_frames = 0;                   // VPADRead calls
uint64_t s_last_update_frame = 0;             // s_vpad_frames at the last camera update
bool s_camera_seen = false;
bool s_last_blocked = false;                  // the last VPADRead dropped the mouse (ESC menu)
// Camera side (under s_lock).
struct Camera {
    uint32_t self = 0;
    float offset = 0;         // degrees the mouse adds to the game's pitch
    bool wrote = false;       // this+0xFC holds our value, not the game's
    float game_fc = 0;        // the game's own value to put back
    uint32_t written_bits = 0; // what we stored, to notice the game changing it
} s_camera;
Debug s_debug;

bool valid_camera(const uint8_t* mem, uint32_t self) {
    if (self < 0x10000000u || (self & 3u) || uint64_t(self) + kObjectSize > WIIU_MEM_SIZE ||
        !rbrew_memory_accessible(mem, self, kObjectSize))
        return false;
    const float hi = rf(mem, self + kPitchMax), mid = rf(mem, self + kPitchMid), lo = rf(mem, self + kPitchMin);
    if (!(hi > mid + 1.0f && mid > lo + 1.0f && hi <= 90.0f && lo >= -90.0f)) return false; // also rejects NaN
    const float x = rf(mem, self + kDir), z = rf(mem, self + kDir + 8);
    const float len = x * x + z * z;
    if (!(len > 0.8f && len < 1.25f)) return false;
    const float fc = rf(mem, self + kPitch);
    return fc >= -1.01f && fc <= 1.01f;
}

void rotate(uint8_t* mem, uint32_t vec, float c, float s) {
    const float x = rf(mem, vec), z = rf(mem, vec + 8);
    if (!std::isfinite(x) || !std::isfinite(z)) return;
    wf(mem, vec, s * z + c * x); // as 0204EFD8: yaw (atan2(x, z)) grows by the angle
    wf(mem, vec + 8, c * z - s * x);
}

void camera_input_thunk(CPUState* cpu) {
    uint8_t* mem = cpu->mem;
    const uint32_t self = cpu->r[3];
    const uint32_t index = kCameraInput >> 2;
    const HostFunc original = index < cpu->dispatch_size ? cpu->dispatch_table[index] : nullptr;
    if (!original) return; // cannot happen: the game registers all its functions
    bool ours = false;
    if (enabled() && valid_camera(mem, self)) {
        std::lock_guard lock(s_lock);
        Camera& cam = s_camera;
        if (cam.self != self) {
            // A second camera in the same frame (another player) is left alone.
            if (!cam.self || s_last_update_frame != s_vpad_frames || !s_camera_seen) cam = Camera{self};
        }
        if (cam.self == self) {
            ours = true;
            if (cam.wrote) {
                // The game's own pitch goes back before its update. If the game
                // rewrote the value meanwhile (camera reset), its value stands
                // and the mouse offset starts again from it.
                if (rbrew_read32(mem, self + kPitch) == cam.written_bits)
                    wf(mem, self + kPitch, cam.game_fc);
                else
                    cam.offset = 0;
                cam.wrote = false;
            }
        }
    } else {
        std::lock_guard lock(s_lock);
        if (s_camera.self == self) s_camera = Camera{};
    }

    original(cpu); // the game's update, unchanged (cpu->lr already returns to 025DB0DC)

    if (!ours || !valid_camera(mem, self)) {
        std::lock_guard lock(s_lock);
        if (ours) s_camera = Camera{};
        if (!ours && !enabled() && valid_camera(mem, self)) {
            // Path off: only note what the game did (diagnostics, autotest).
            Debug& d = s_debug;
            const float hi = rf(mem, self + kPitchMax), mid = rf(mem, self + kPitchMid), lo = rf(mem, self + kPitchMin);
            const float fc = rf(mem, self + kPitch);
            d = Debug{self, d.updates + 1, rbrew_read32(mem, self + kLocked) != 0};
            d.yaw = std::atan2(rf(mem, self + kDirSmooth), rf(mem, self + kDirSmooth + 8)) * (180.0f / kPi);
            d.pitch = d.game_pitch = fc >= 0 ? mid + (hi - mid) * fc : mid + (mid - lo) * fc;
            d.pitch_max = hi;
            d.pitch_mid = mid;
            d.pitch_min = lo;
            d.vpad_frame = s_vpad_frames;
        }
        return;
    }
    std::lock_guard lock(s_lock);
    Camera& cam = s_camera;
    s_last_update_frame = s_vpad_frames;
    s_camera_seen = true;
    float yaw_step = s_pending_yaw, pitch_step = s_pending_pitch;
    s_pending_yaw = s_pending_pitch = 0;
    const float weight = rf(mem, self + kInputWeight);
    // The game holds the camera, or the player has no control of it (title
    // screen, cinematics, respawn/super-jump lock): drop mouse movement, clear
    // any stale pitch offset, and leave the game's pitch untouched.
    const bool locked = rbrew_read32(mem, self + kLocked) != 0 || !(weight > 0.0f);
    if (locked) {
        yaw_step = pitch_step = 0;
        cam.offset = 0;
    }

    if (yaw_step != 0) {
        const float angle = yaw_step * (kPi / 180.0f);
        const float c = std::cos(angle), s = std::sin(angle);
        rotate(mem, self + kDir, c, s);
        rotate(mem, self + kDirSmooth, c, s);
        const float stick_yaw = rf(mem, self + kStickYaw);
        if (std::isfinite(stick_yaw)) wf(mem, self + kStickYaw, stick_yaw + angle);
    }

    const float hi = rf(mem, self + kPitchMax), mid = rf(mem, self + kPitchMid), lo = rf(mem, self + kPitchMin);
    const float game_fc = rf(mem, self + kPitch);
    const float game_pitch = game_fc >= 0 ? mid + (hi - mid) * game_fc : mid + (mid - lo) * game_fc;
    float pitch = game_pitch;
    if (!locked) {
        pitch = game_pitch + cam.offset + pitch_step;
        if (pitch > hi) pitch = hi;
        if (pitch < lo) pitch = lo;
        cam.offset = pitch - game_pitch;
        // The stick holds the game's pitch on a limit the offset keeps the camera
        // away from: the offset gives way, so the stick alone reaches every angle.
        if (s_stick_pitch && pitch_step == 0 && cam.offset != 0 && std::fabs(game_fc) > 0.999f &&
            (game_fc > 0) != (cam.offset > 0)) {
            constexpr float kGiveWay = 3.0f; // degrees per frame
            cam.offset = std::fabs(cam.offset) <= kGiveWay ? 0.0f : cam.offset - std::copysign(kGiveWay, cam.offset);
            pitch = game_pitch + cam.offset;
        }
        if (std::fabs(cam.offset) < 1e-4f) {
            cam.offset = 0;
            pitch = game_pitch;
        } else {
            float fc = pitch >= mid ? (pitch - mid) / (hi - mid) : (pitch - mid) / (mid - lo);
            if (fc > 1.0f) fc = 1.0f;
            if (fc < -1.0f) fc = -1.0f;
            wf(mem, self + kPitch, fc);
            cam.game_fc = game_fc;
            cam.written_bits = rbrew_read32(mem, self + kPitch);
            cam.wrote = true;
        }
    }

    Debug& d = s_debug;
    d.camera = self;
    ++d.updates;
    d.locked = locked;
    d.yaw = std::atan2(rf(mem, self + kDirSmooth), rf(mem, self + kDirSmooth + 8)) * (180.0f / kPi);
    d.pitch = pitch;
    d.game_pitch = game_pitch;
    d.offset = cam.offset;
    d.pitch_max = hi;
    d.pitch_mid = mid;
    d.pitch_min = lo;
    d.applied_yaw = yaw_step;
    d.applied_pitch = pitch_step;
    d.vpad_frame = s_vpad_frames;
    d.weight = weight;
}
} // namespace

void install(CPUState* cpu) {
    // Always bound: the patched call has no other target. With the path off
    // the thunk only forwards to the game's function.
    rbrew_register_func(cpu, kThunk, camera_input_thunk);
}

bool enabled() {
    return rbrew_settings::g_mouse_direct.load(std::memory_order_relaxed) &&
           rbrew_settings::g_mouse_camera.load(std::memory_order_relaxed);
}

bool vpad_frame(float yaw_deg, float pitch_deg, float stick_pitch, bool blocked) {
    std::lock_guard lock(s_lock);
    ++s_vpad_frames;
    s_last_blocked = blocked;
    s_stick_pitch = std::fabs(stick_pitch) > 0.3f;
    // A camera update in the last few polls: the player camera is running.
    const bool alive = s_camera_seen && s_vpad_frames - s_last_update_frame <= 4;
    if (!alive) {
        s_camera_seen = false;
        s_pending_yaw = s_pending_pitch = 0;
        // The object may be freed (scene change): never put a remembered
        // pitch back into whatever reuses its address.
        s_camera = Camera{};
    }
    if (!enabled() || !alive) return false;
    if (blocked || !std::isfinite(yaw_deg) || !std::isfinite(pitch_deg)) {
        s_pending_yaw = s_pending_pitch = 0;
        return true;
    }
    s_pending_yaw += yaw_deg;
    s_pending_pitch += pitch_deg;
    return true;
}

Debug debug() {
    std::lock_guard lock(s_lock);
    return s_debug;
}

// ---------------------------------------------------------------------------
// RBREW_AIM_AUTOTEST (development)
// ---------------------------------------------------------------------------
namespace {
constexpr uint32_t kBtnA = 0x8000u, kBtnZL = 0x80u, kBtnZR = 0x40u;
enum Kind { kIdle, kMouse, kFlick, kZigzag, kStick, kWalk, kShoot, kBoth };
struct Segment {
    int id, frames, kind;
    float a, b; // mouse counts per frame / stick deflection
    int every;  // flick period; kShoot: frame of a photo (Screenshots folder)
    float rx = 0, ry = 0; // kBoth: the stick with the mouse
};
// id < 100: mouse traces; 100..199: controller only; 200+: mouse while walking
// or shooting.
constexpr Segment kSegments[] = {
    {1, 60, kIdle, 0, 0, 0},
    {2, 300, kMouse, 2, 0, 0},     // slow sweep right
    {3, 300, kMouse, -2, 0, 0},    // and back
    {4, 30, kIdle, 0, 0, 0},
    {5, 120, kFlick, 60, 0, 20},   // small fast flicks
    {6, 120, kFlick, -60, 0, 20},
    {7, 121, kZigzag, 1, 0, 0},    // fast back-and-forth, sums to zero
    {8, 60, kIdle, 0, 0, 0},
    {9, 400, kMouse, 0, -3, 0},    // up past the limit
    {10, 700, kMouse, 0, 3, 0},    // down past the limit
    {11, 281, kMouse, 0, -3, 0},   // back to level
    {12, 30, kIdle, 0, 0, 0},
    {13, 60, kFlick, 0, -40, 15},
    {14, 60, kFlick, 0, 40, 15},
    {15, 121, kZigzag, 1, 1, 0},   // both axes
    {16, 120, kMouse, 5, -2, 0},   // diagonal
    {17, 120, kMouse, -5, 2, 0},
    {18, 120, kIdle, 0, 0, 0},
    {101, 90, kStick, 0.6f, 0, 0}, // controller path, mouse idle
    {102, 60, kIdle, 0, 0, 0},
    {103, 90, kStick, -0.6f, 0, 0},
    {104, 60, kIdle, 0, 0, 0},
    {105, 60, kStick, 0, 0.6f, 0},
    {106, 40, kIdle, 0, 0, 0},
    {107, 60, kStick, 0, -0.6f, 0},
    {108, 120, kIdle, 0, 0, 0},
    {201, 90, kWalk, 3, 0, 0},     // mouse while walking forward
    {202, 90, kIdle, 0, 0, 0},
    {203, 120, kShoot, 4, 0, 0},   // mouse while shooting: the shots follow
    {204, 60, kShoot, 0, 0, 0},
    {205, 120, kShoot, -4, -1, 0},
    {206, 60, kShoot, 0, 0, 0},
    {207, 60, kIdle, 0, 0, 0},
    {208, 40, kMouse, 0, -13, 0},  // look up ~40 degrees and shoot (photo)
    {209, 60, kShoot, 0, 0, 50},
    {210, 80, kMouse, 0, 13, 0},   // look down and shoot (photo)
    {211, 60, kShoot, 0, 0, 50},
    {212, 40, kMouse, 0, -13, 0},
    {213, 60, kIdle, 0, 0, 0},
    {220, 30, kMouse, 0, -8, 0},   // mouse pitch offset, then the stick pitches
    {221, 60, kStick, 0, -0.6f, 0},
    {222, 60, kIdle, 0, 0, 0},
    {223, 60, kStick, 0, 0.6f, 0},
    {224, 60, kIdle, 0, 0, 0},
    {225, 60, kBoth, 5, 0, 0, 0.3f, 0}, // stick and mouse together
    {226, 60, kIdle, 0, 0, 0},
};

// RBREW_AIM_AUTOTEST_MODE=look: turn 45 degrees and wait, eight times (photos
// of the surroundings, to find the way somewhere).
constexpr Segment kLook[] = {
    {301, 31, kMouse, -18, 0, 0},
    {302, 120, kIdle, 0, 0, 0},
};

// RBREW_AIM_AUTOTEST_MODE=shoot: aim up, down, left with the mouse and shoot
// for four seconds each (window pictures show where the ink goes).
constexpr Segment kShootPlan[] = {
    {401, 40, kMouse, 0, -13, 0}, {402, 240, kShoot, 0, 0, 0}, {403, 80, kMouse, 0, 13, 0},
    {404, 240, kShoot, 0, 0, 0},  {405, 40, kMouse, 0, -13, 0}, {406, 30, kMouse, -25, 0, 0},
    {407, 240, kShoot, 0, 0, 0},  {408, 60, kIdle, 0, 0, 0},
};

struct AutoTest {
    FILE* log = nullptr;
    int scene = -1;       // scene kind to wait for (-1: any scene with a player camera)
    int mode = 0;         // 0 all, 1 mouse only, 2 controller only
    bool navigate = false; // press ZL+ZR / A until the player camera is in control
    uint64_t n = 0;
    int matched = 0;      // consecutive frames the picture followed the player camera
    bool ready = false;
    size_t segment = 0;
    int frame_in_segment = 0;
    uint32_t rng = 12345u;
    int zig_sum_x = 0, zig_sum_y = 0;
    bool done = false;
    int photo_every = 0; // navigation: a photo every N frames (Screenshots folder)
    int title_wait = 0;
    int motion = -1; // RBREW_AIM_AUTOTEST_MOTION=0/1: force the game's motion controls option
    uint64_t first_camera = 0;
    bool nav_mouse = false; // RBREW_AIM_AUTOTEST_NAVMOUSE=1: mouse noise while navigating (menus, title)
    int repeat = 1;      // RBREW_AIM_AUTOTEST_REPEAT: play the whole list N times
    uint64_t last_updates = 0;
    float probe_x = 0, probe_z = 0; // eye at the start of a walk probe
};

AutoTest* autotest() {
    static AutoTest* test = []() -> AutoTest* {
        const char* path = std::getenv("RBREW_AIM_AUTOTEST");
        if (!path || !*path) return nullptr;
        FILE* f = std::fopen(path, "w");
        if (!f) return nullptr;
        auto* t = new AutoTest;
        t->log = f;
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_SCENE")) t->scene = std::atoi(v);
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_MODE"))
            t->mode = !std::strcmp(v, "mouse") ? 1 : !std::strcmp(v, "stick") ? 2 : !std::strcmp(v, "look") ? 3 : !std::strcmp(v, "shoot") ? 4 : 0;
        if (t->mode == 3) t->repeat = 8;
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_NAV")) t->navigate = v[0] != '0';
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_PHOTO")) t->photo_every = std::atoi(v);
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_TITLEWAIT")) t->title_wait = std::atoi(v);
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_MOTION")) t->motion = std::atoi(v) ? 1 : 0;
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_NAVMOUSE")) t->nav_mouse = v[0] == '1';
        if (const char* v = std::getenv("RBREW_AIM_AUTOTEST_REPEAT")) t->repeat = std::max(1, std::atoi(v));
        std::fprintf(f, "n seg scene scene_id buttons mdx mdy rx ry lx ly motion view_ok view_yaw view_pitch updates "
                        "upd_vpad cam_yaw cam_pitch game_pitch offset pmax pmid pmin locked applied_yaw applied_pitch "
                        "pose_yaw pose_pitch player_yaw camera eye_x eye_z flags weight\n");
        return t;
    }();
    return test;
}

const Segment* plan(const AutoTest& t) {
    return t.mode == 3 ? kLook : t.mode == 4 ? kShootPlan : kSegments;
}
size_t plan_size(const AutoTest& t) {
    return t.mode == 3 ? std::size(kLook) : t.mode == 4 ? std::size(kShootPlan) : std::size(kSegments);
}
bool segment_wanted(const AutoTest& t, const Segment& s) {
    return t.mode == 0 || t.mode >= 3 || (t.mode == 1 && (s.id < 100 || s.id >= 200)) || (t.mode == 2 && s.id >= 100 && s.id < 200);
}

// The direction the camera's pose (025D7CC0) left in the object: eye at
// +0x58, look-at point at +0x48, read on the next VPADRead, i.e. after the update that
// last ran (same frame as Debug).
bool pose_angles(const uint8_t* mem, uint32_t self, float& yaw, float& pitch) {
    if (!self || !rbrew_memory_accessible(mem, self, kObjectSize)) return false;
    float d[3];
    for (int i = 0; i < 3; ++i) d[i] = rf(mem, self + 0x48 + 4 * i) - rf(mem, self + 0x58 + 4 * i); // look-at - eye
    const float h = std::sqrt(d[0] * d[0] + d[2] * d[2]);
    if (!std::isfinite(h) || !(h + std::fabs(d[1]) > 1e-4f)) return false;
    yaw = std::atan2(d[0], d[2]) * (180.0f / kPi);
    pitch = std::atan2(d[1], h) * (180.0f / kPi);
    return true;
}
// The character's facing (player object, read by the pose at +0x198/+0x1A0).
bool player_yaw(const uint8_t* mem, uint32_t self, float& yaw) {
    if (!self || !rbrew_memory_accessible(mem, self, 4)) return false;
    const uint32_t player = rbrew_read32(const_cast<uint8_t*>(mem), self);
    if (player < 0x10000000u || (player & 3u) || uint64_t(player) + 0x1A4 > WIIU_MEM_SIZE ||
        !rbrew_memory_accessible(mem, player, 0x1A4))
        return false;
    const float x = rf(mem, player + 0x198), z = rf(mem, player + 0x1A0);
    if (!std::isfinite(x) || !std::isfinite(z) || x * x + z * z < 1e-6f) return false;
    yaw = std::atan2(x, z) * (180.0f / kPi);
    return true;
}
// Player object bytes that make 025D4A2C drop the stick: +0x110, +0x145,
// +0x146, +0x147 (one byte each, packed high to low).
uint32_t player_flags(const uint8_t* mem, uint32_t self) {
    if (!self || !rbrew_memory_accessible(mem, self, 4)) return 0xFFFFFFFFu;
    const uint32_t player = rbrew_read32(const_cast<uint8_t*>(mem), self);
    if (player < 0x10000000u || uint64_t(player) + 0x148 > WIIU_MEM_SIZE || !rbrew_memory_accessible(mem, player, 0x148))
        return 0xFFFFFFFFu;
    return uint32_t(mem[player + 0x110]) << 24 | uint32_t(mem[player + 0x145]) << 16 | uint32_t(mem[player + 0x146]) << 8 |
           mem[player + 0x147];
}
// RBREW_AIM_AUTOTEST_DUMP=<prefix>: the player and camera objects, raw.
void dump_objects(const uint8_t* mem, uint32_t self, const char* tag) {
    const char* prefix = std::getenv("RBREW_AIM_AUTOTEST_DUMP");
    if (!prefix || !*prefix || !self || !rbrew_memory_accessible(mem, self, kObjectSize)) return;
    const uint32_t player = rbrew_read32(const_cast<uint8_t*>(mem), self);
    char path[512];
    std::snprintf(path, sizeof(path), "%s_%s_camera_%08X.bin", prefix, tag, self);
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(mem + self, 1, kObjectSize, f);
        std::fclose(f);
    }
    if (player >= 0x10000000u && uint64_t(player) + 0x1000 <= WIIU_MEM_SIZE && rbrew_memory_accessible(mem, player, 0x1000)) {
        std::snprintf(path, sizeof(path), "%s_%s_player_%08X.bin", prefix, tag, player);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(mem + player, 1, 0x1000, f);
            std::fclose(f);
        }
    }
}
int motion_option(const uint8_t* mem) {
    const uint32_t manager = rbrew_read32(const_cast<uint8_t*>(mem), RBREW_GAME_CONTROL_MANAGER);
    if (!manager || uint64_t(manager) + 0x184 > WIIU_MEM_SIZE || !rbrew_memory_accessible(mem, manager + 0x17C, 8))
        return -1;
    return mem[manager + 0x182];
}
} // namespace

void smooth_feed(TestPad& pad);
bool autotest_frame(uint8_t* mem, TestPad& pad) {
    AutoTest* t = autotest();
    if (!t) return false;
    pad = TestPad{};
    ++t->n;
    if (t->motion >= 0) {
        // After vpad_apply_motion_policy (earlier in this VPADRead), before the game reads it.
        const uint32_t manager = rbrew_read32(mem, RBREW_GAME_CONTROL_MANAGER);
        if (manager && uint64_t(manager) + 0x184 <= WIIU_MEM_SIZE && rbrew_memory_accessible(mem, manager + 0x17C, 8))
            rbrew_write8(mem, manager + 0x182, uint8_t(t->motion));
    }
    int scene_id = -1;
    const int scene = rbrew_control_defaults::active_scene_type(mem, &scene_id);
    const Debug d = debug();
    if (t->photo_every > 0 && t->n % uint64_t(t->photo_every) == 0) gx2_render_request_photo();
    gx2env::MainView view{};
    float view_yaw = 0, view_pitch = 0;
    const bool have_view = gx2env::main_view(view);
    if (have_view) {
        const float dx = view.at[0] - view.pos[0], dy = view.at[1] - view.pos[1], dz = view.at[2] - view.pos[2];
        view_yaw = std::atan2(dx, dz) * (180.0f / kPi);
        view_pitch = std::atan2(dy, std::sqrt(dx * dx + dz * dz)) * (180.0f / kPi);
    }
    float pose_yaw = 0, pose_pitch = 0, body_yaw = 0;
    const bool have_pose = pose_angles(mem, d.camera, pose_yaw, pose_pitch);
    player_yaw(mem, d.camera, body_yaw);
    const bool alive = d.updates != t->last_updates;
    t->last_updates = d.updates;
    int segment_id = 0;
    if (!t->ready) {
        // The player is in control when walking moves the camera: the title
        // screen also runs a player camera over the plaza, but ignores the
        // sticks. Every 120 frames: buttons (title, messages), then a walk
        // probe of 40 frames.
        const bool usable = (t->scene < 0 || scene == t->scene) && alive && have_pose && !d.locked;
        const int k = int(t->n % 120);
        if (t->nav_mouse) {
            pad.mdx = (t->n / 7) % 2 ? 9 : -9;
            pad.mdy = (t->n / 11) % 2 ? 6 : -6;
        }
        // RBREW_AIM_AUTOTEST_TITLEWAIT=<frames>: once a player camera runs,
        // wait that long before pressing anything (the title screen stays).
        if (d.updates && !t->first_camera) t->first_camera = t->n;
        const bool title_wait = t->first_camera && t->n - t->first_camera < uint64_t(t->title_wait);
        if (title_wait) {
            pad.rx = 0.5f;
            pad.mdx = 4;
            if (t->n - t->first_camera == uint64_t(t->title_wait) / 2) dump_objects(mem, d.camera, "title");
            if (t->n - t->first_camera == uint64_t(t->title_wait) / 4) dump_objects(mem, d.camera, "title0");
        } else if (k < 80) {
            if (t->navigate && k % 40 < 8) pad.buttons = kBtnZL | kBtnZR; // title screen
            else if (t->navigate && k % 40 >= 20 && k % 40 < 28) pad.buttons = kBtnA; // messages
        } else {
            pad.ly = 1.0f;
            const float x = have_pose ? rf(mem, d.camera + 0x58) : 0.0f, z = have_pose ? rf(mem, d.camera + 0x60) : 0.0f;
            if (k == 80) {
                t->probe_x = x;
                t->probe_z = z;
            } else if (k == 119) {
                const float moved = std::sqrt((x - t->probe_x) * (x - t->probe_x) + (z - t->probe_z) * (z - t->probe_z));
                t->matched = usable && std::isfinite(moved) && moved > 3.0f ? t->matched + 1 : 0;
                std::fprintf(t->log, "PROBE moved %.2f usable %d\n", moved, usable ? 1 : 0);
            }
        }
        if (t->matched >= 2 && k == 119) {
            t->ready = true;
            std::fprintf(t->log, "READY\n");
            dump_objects(mem, d.camera, "ready");
            while (t->segment < plan_size(*t) && !segment_wanted(*t, plan(*t)[t->segment])) ++t->segment;
        }
        segment_id = -t->matched;
    } else if (t->segment < plan_size(*t)) {
        const Segment& s = plan(*t)[t->segment];
        segment_id = s.id;
        const int f = t->frame_in_segment;
        switch (s.kind) {
        case kMouse:
            pad.mdx = int(s.a);
            pad.mdy = int(s.b);
            break;
        case kFlick:
            if (f % s.every == 0) {
                pad.mdx = int(s.a);
                pad.mdy = int(s.b);
            }
            break;
        case kZigzag: {
            if (f == 0) t->zig_sum_x = t->zig_sum_y = 0;
            if (f == s.frames - 1) { // the last frame returns to the start
                pad.mdx = -t->zig_sum_x;
                pad.mdy = -t->zig_sum_y;
            } else {
                t->rng = t->rng * 1664525u + 1013904223u;
                const int mx = 20 + int((t->rng >> 16) % 101u);
                t->rng = t->rng * 1664525u + 1013904223u;
                const int my = 5 + int((t->rng >> 16) % 30u);
                const int sign = (f & 1) ? -1 : 1;
                if (s.a != 0) pad.mdx = sign * mx;
                if (s.b != 0) pad.mdy = -sign * my;
                t->zig_sum_x += pad.mdx;
                t->zig_sum_y += pad.mdy;
            }
            break;
        }
        case kStick:
            if (s.id == 101 && f == 45) dump_objects(mem, d.camera, "stick");
            pad.rx = s.a;
            pad.ry = s.b;
            break;
        case kWalk:
            pad.ly = 1.0f;
            pad.mdx = int(s.a);
            break;
        case kBoth:
            pad.rx = s.rx;
            pad.ry = s.ry;
            pad.mdx = int(s.a);
            pad.mdy = int(s.b);
            break;
        case kShoot:
            if (s.every && f == s.every) gx2_render_request_photo();
            pad.buttons = kBtnZR;
            pad.mdx = int(s.a);
            pad.mdy = int(s.b);
            break;
        default: break;
        }
        if (++t->frame_in_segment >= s.frames) {
            t->frame_in_segment = 0;
            do ++t->segment;
            while (t->segment < plan_size(*t) && !segment_wanted(*t, plan(*t)[t->segment]));
            if (t->segment >= plan_size(*t) && --t->repeat > 0) {
                t->segment = 0;
                while (t->segment < plan_size(*t) && !segment_wanted(*t, plan(*t)[t->segment])) ++t->segment;
            }
        }
    } else {
        segment_id = 999;
        if (!t->done) {
            t->done = true;
            dump_objects(mem, d.camera, "done");
            std::fprintf(t->log, "DONE\n");
            std::fflush(t->log);
        }
    }
    smooth_feed(pad);
    if (!t->done || (t->n & 63) == 0) {
        std::fprintf(t->log,
                     "%llu %d %d %d %X %d %d %.3f %.3f %.3f %.3f %d %d %.4f %.4f %llu %llu %.4f %.4f %.4f %.4f %.3f %.3f "
                     "%.3f %d %.4f %.4f %.4f %.4f %.4f %08X %.2f %.2f %08X %.3f\n",
                     (unsigned long long)t->n, segment_id, scene, scene_id, pad.buttons, pad.mdx, pad.mdy, pad.rx, pad.ry,
                     pad.lx, pad.ly, motion_option(mem), have_view ? 1 : 0, view_yaw, view_pitch,
                     (unsigned long long)d.updates, (unsigned long long)d.vpad_frame, d.yaw, d.pitch, d.game_pitch,
                     d.offset, d.pitch_max, d.pitch_mid, d.pitch_min, d.locked ? 1 : 0, d.applied_yaw, d.applied_pitch,
                     have_pose ? pose_yaw : 999.0f, have_pose ? pose_pitch : 999.0f, body_yaw, d.camera,
                     have_pose ? rf(mem, d.camera + 0x58) : 0.0f, have_pose ? rf(mem, d.camera + 0x60) : 0.0f, player_flags(mem, d.camera), d.weight);
        if ((t->n & 31) == 0) std::fflush(t->log);
    }
    return true;
}
// ---------------------------------------------------------------------------
// Mouse look at the display's rate (see mouse_aim.h)
// ---------------------------------------------------------------------------
namespace {
std::mutex s_count_lock;
int64_t s_taken_x = 0, s_taken_y = 0;         // counts given to the game (VPADRead)
uint64_t s_taken_ns = 0;                      // steady-clock ns of the last take_counts
int64_t s_raw_x = 0, s_raw_y = 0;             // counts the raw thread received
int64_t s_raw_taken_x = 0, s_raw_taken_y = 0; // of those, given to the game
struct MouseSample {
    uint64_t ns = 0;
    int64_t x = 0, y = 0;
};
// 1024 samples covers >1 s at 1000 Hz and 128..256 ms on 4000-8000 Hz mice
// without clamping counts_at() during stage hitches.
constexpr size_t kMouseHist = 1024;
MouseSample s_hist[kMouseHist]{};
size_t s_hist_head = 0, s_hist_count = 0;
uint64_t aim_now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
}
void push_sample_locked(uint64_t ns) {
    const int64_t tx = s_taken_x + (s_raw_x - s_raw_taken_x);
    const int64_t ty = s_taken_y + (s_raw_y - s_raw_taken_y);
    if (s_hist_count > 0) {
        MouseSample& prev = s_hist[(s_hist_head + kMouseHist - 1) % kMouseHist];
        if (prev.ns >= ns) {
            prev.x = tx;
            prev.y = ty;
            return;
        }
    }
    s_hist[s_hist_head] = {ns, tx, ty};
    s_hist_head = (s_hist_head + 1) % kMouseHist;
    if (s_hist_count < kMouseHist) ++s_hist_count;
}
std::atomic<float> s_degrees_per_count{0.0f};
std::atomic<bool> s_relative{false};
std::atomic<bool> s_raw_owns{false};
std::atomic<uint32_t> s_raw_buttons{0}; // held SDL_BUTTON(1..5) bitmask from WM_INPUT
std::atomic<uint32_t> s_raw_pressed{0}; // latched presses since last take_buttons (sub-frame taps)
bool s_raw_started = false; // game thread

void add_raw(int64_t dx, int64_t dy) {
    const uint64_t ns = aim_now_ns();
    std::lock_guard lock(s_count_lock);
    s_raw_x += dx;
    s_raw_y += dy;
    push_sample_locked(ns);
}

// RBREW_RAW_MOUSE=0: no raw input thread (SDL only: the in-between frames
// then see the mouse once per step). Never in test windows (they must not
// take input meant for whoever uses the PC).
bool raw_wanted() {
    static const bool on = [] {
        const char* v = std::getenv("RBREW_RAW_MOUSE");
        if (v && v[0] == '0') return false;
        if (const char* t = std::getenv("RBREW_TEST_WINDOW"); t && *t && t[0] != '0') return false;
        if (const char* g = std::getenv("RBREW_NO_MOUSE_GRAB"); g && g[0] == '1') return false;
        return true;
    }();
    return on;
}

#if defined(_WIN32)
// The raw input thread. Windows gives each process one raw-input target per
// device class, and SDL's relative mode registers the mouse to the game
// window, whose messages only the game thread reads (once per step). This
// thread registers its own message-only window instead (RIDEV_INPUTSINK) while
// relative mode and high-FPS replay are active, counting movement and tracking
// buttons as they arrive (while SDL's relative mode is on, SDL2 ignores legacy
// WM_*BUTTON* and expects buttons via WM_INPUT, so this thread relays them to
// vpad.cpp via take_buttons). When relative mode or high-FPS mode turns off
// (ESC menu, GamePad view, 60 FPS), the thread unregisters so SDL/ImGui keep
// clean input ownership.
HWND s_raw_window = nullptr;
std::atomic<HWND> s_raw_window_atomic{nullptr};
DWORD s_pid = 0;
bool raw_want_active() {
    return s_relative.load(std::memory_order_relaxed) && raw_wanted() && enabled() &&
           rbrew_settings::g_fps120.load(std::memory_order_relaxed) &&
           !rbrew_settings::g_frame_gen.load(std::memory_order_relaxed);
}
bool raw_counting() {
    if (!s_relative.load(std::memory_order_relaxed)) return false;
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    if (fg) GetWindowThreadProcessId(fg, &pid);
    return pid == s_pid;
}
void seed_raw_buttons() {
    uint32_t mask = 0;
    if (raw_counting()) {
        // GetAsyncKeyState checks physical buttons, matching RI_MOUSE_LEFT/RIGHT.
        const bool swapped = GetSystemMetrics(SM_SWAPBUTTON) != 0;
        const uint32_t left_bit = swapped ? (1u << 2) : (1u << 0);
        const uint32_t right_bit = swapped ? (1u << 0) : (1u << 2);
        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) mask |= left_bit;
        if (GetAsyncKeyState(VK_MBUTTON) & 0x8000) mask |= 1u << 1;
        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) mask |= right_bit;
        if (GetAsyncKeyState(VK_XBUTTON1) & 0x8000) mask |= 1u << 3;
        if (GetAsyncKeyState(VK_XBUTTON2) & 0x8000) mask |= 1u << 4;
    }
    s_raw_buttons.store(mask, std::memory_order_release);
}
void update_raw_buttons(USHORT flags, bool active) {
    const bool swapped = GetSystemMetrics(SM_SWAPBUTTON) != 0;
    const uint32_t left_bit = swapped ? (1u << 2) : (1u << 0);
    const uint32_t right_bit = swapped ? (1u << 0) : (1u << 2);
    uint32_t down = 0, up = 0;
    if (flags & RI_MOUSE_LEFT_BUTTON_DOWN) down |= left_bit;
    if (flags & RI_MOUSE_LEFT_BUTTON_UP) up |= left_bit;
    if (flags & RI_MOUSE_RIGHT_BUTTON_DOWN) down |= right_bit;
    if (flags & RI_MOUSE_RIGHT_BUTTON_UP) up |= right_bit;
    if (flags & RI_MOUSE_MIDDLE_BUTTON_DOWN) down |= 1u << 1;
    if (flags & RI_MOUSE_MIDDLE_BUTTON_UP) up |= 1u << 1;
    if (flags & RI_MOUSE_BUTTON_4_DOWN) down |= 1u << 3;
    if (flags & RI_MOUSE_BUTTON_4_UP) up |= 1u << 3;
    if (flags & RI_MOUSE_BUTTON_5_DOWN) down |= 1u << 4;
    if (flags & RI_MOUSE_BUTTON_5_UP) up |= 1u << 4;
    // Apply DOWN before UP: if a single packet carries both, latch the tap in
    // s_raw_pressed while leaving s_raw_buttons released (never stuck down).
    if (down && active) {
        s_raw_buttons.fetch_or(down, std::memory_order_acq_rel);
        s_raw_pressed.fetch_or(down, std::memory_order_acq_rel);
    }
    if (up) s_raw_buttons.fetch_and(~up, std::memory_order_acq_rel);
}
bool raw_registered() {
    RAWINPUTDEVICE devices[16];
    UINT count = 16;
    const UINT got = GetRegisteredRawInputDevices(devices, &count, sizeof(RAWINPUTDEVICE));
    if (got == UINT(-1)) return true; // more than 16 classes: leave it alone
    for (UINT i = 0; i < got; ++i)
        if (devices[i].usUsagePage == 1 && devices[i].usUsage == 2) return devices[i].hwndTarget == s_raw_window;
    return false;
}
void raw_register() {
    RAWINPUTDEVICE device{1, 2, RIDEV_INPUTSINK, s_raw_window};
    RegisterRawInputDevices(&device, 1, sizeof(device));
}
void raw_unregister() {
    RAWINPUTDEVICE device{1, 2, RIDEV_REMOVE, nullptr};
    RegisterRawInputDevices(&device, 1, sizeof(device));
}
void raw_sync_registration() {
    if (raw_want_active()) {
        if (!raw_registered()) {
            raw_register();
            if (raw_registered()) {
                seed_raw_buttons();
                s_raw_owns.store(true, std::memory_order_release);
            } else {
                s_raw_owns.store(false, std::memory_order_release);
            }
        } else if (!s_raw_owns.load(std::memory_order_relaxed)) {
            seed_raw_buttons();
            s_raw_owns.store(true, std::memory_order_release);
        } else if (!raw_counting()) {
            s_raw_buttons.store(0, std::memory_order_release);
            s_raw_pressed.store(0, std::memory_order_release);
        }
    } else {
        s_raw_owns.store(false, std::memory_order_release);
        s_raw_buttons.store(0, std::memory_order_release);
        s_raw_pressed.store(0, std::memory_order_release);
        if (raw_registered()) raw_unregister();
    }
}
LRESULT CALLBACK raw_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (msg == WM_INPUT) {
        RAWINPUT input;
        UINT size = sizeof(input);
        if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lparam), RID_INPUT, &input, &size, sizeof(RAWINPUTHEADER)) !=
                UINT(-1) &&
            input.header.dwType == RIM_TYPEMOUSE) {
            const RAWMOUSE& m = input.data.mouse;
            const bool active = raw_counting();
            if (!active) {
                s_raw_buttons.store(0, std::memory_order_release);
                s_raw_pressed.store(0, std::memory_order_release);
            } else {
                if (!(m.usFlags & MOUSE_MOVE_ABSOLUTE) && (m.lLastX || m.lLastY))
                    add_raw(m.lLastX, m.lLastY);
                if (m.usButtonFlags) update_raw_buttons(m.usButtonFlags, true);
            }
        }
        return 0;
    }
    if (msg == WM_APP) {
        raw_sync_registration();
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}
void raw_thread() {
    s_pid = GetCurrentProcessId();
    WNDCLASSW wc{};
    wc.lpfnWndProc = raw_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"RebrewRawMouse";
    RegisterClassW(&wc);
    s_raw_window = CreateWindowExW(0, wc.lpszClassName, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
    if (!s_raw_window) return;
    s_raw_window_atomic.store(s_raw_window, std::memory_order_release);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    raw_sync_registration();
    ULONGLONG last_check = GetTickCount64();
    for (;;) {
        MsgWaitForMultipleObjects(0, nullptr, FALSE, 50, QS_ALLINPUT);
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        const ULONGLONG now = GetTickCount64();
        if (now - last_check >= 50) {
            last_check = now;
            raw_sync_registration();
        }
    }
}
#endif

// RBREW_AIM_SMOOTH=1 with RBREW_AIM_AUTOTEST (tests of the display-rate
// look): the autotest's mouse traces are not given per frame but fed into
// the raw counters at a steady speed by a thread (2 kHz), like a real mouse.
std::atomic<float> s_smooth_vx{0.0f}, s_smooth_vy{0.0f}; // counts per second
bool s_smooth_started = false;
bool smooth_test() {
    static const bool on = [] {
        const char* v = std::getenv("RBREW_AIM_SMOOTH");
        return v && v[0] == '1' && std::getenv("RBREW_AIM_AUTOTEST");
    }();
    return on;
}
void smooth_thread() {
    using clock = std::chrono::steady_clock;
    auto last = clock::now();
    double fx = 0, fy = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        const auto now = clock::now();
        const double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        fx += s_smooth_vx.load(std::memory_order_relaxed) * dt;
        fy += s_smooth_vy.load(std::memory_order_relaxed) * dt;
        const int64_t ix = int64_t(fx), iy = int64_t(fy);
        if (ix || iy) {
            fx -= double(ix);
            fy -= double(iy);
            add_raw(ix, iy);
        }
    }
}
} // namespace

void smooth_feed(TestPad& pad) {
    if (!smooth_test()) return;
    if (!s_smooth_started) {
        s_smooth_started = true;
        std::thread(smooth_thread).detach();
    }
    // One frame's counts spread over the next 1/59.94 s.
    s_smooth_vx.store(float(pad.mdx) * 59.94f, std::memory_order_relaxed);
    s_smooth_vy.store(float(pad.mdy) * 59.94f, std::memory_order_relaxed);
    pad.mdx = pad.mdy = 0;
}

void take_counts(int sdl_dx, int sdl_dy, bool relative, int& dx, int& dy) {
    const bool prev_rel = s_relative.exchange(relative, std::memory_order_relaxed);
    if (!relative) {
        s_raw_buttons.store(0, std::memory_order_relaxed);
        s_raw_pressed.store(0, std::memory_order_relaxed);
    }
#if defined(_WIN32)
    static bool s_last_want = false;
    const bool want = raw_want_active();
    if (!s_raw_started && want) {
        s_raw_started = true;
        std::thread(raw_thread).detach();
    } else if (s_raw_started && (prev_rel != relative || s_last_want != want)) {
        if (HWND w = s_raw_window_atomic.load(std::memory_order_acquire)) PostMessageW(w, WM_APP, 0, 0);
    }
    s_last_want = want;
#else
    (void)prev_rel;
#endif
    const uint64_t ns = aim_now_ns();
    std::lock_guard lock(s_count_lock);
    const int64_t rx = s_raw_x - s_raw_taken_x, ry = s_raw_y - s_raw_taken_y;
    s_raw_taken_x = s_raw_x;
    s_raw_taken_y = s_raw_y;
    dx = int(std::clamp<int64_t>(int64_t(sdl_dx) + rx, -1000000, 1000000));
    dy = int(std::clamp<int64_t>(int64_t(sdl_dy) + ry, -1000000, 1000000));
    s_taken_x += dx;
    s_taken_y += dy;
    s_taken_ns = ns;
    // When the raw thread owns mouse input and SDL added no movement, avoid
    // injecting a fake zero-velocity knot between hardware WM_INPUT packets
    // (which would create a 60 Hz velocity ripple in counts_at()); still anchor
    // idle pauses (>12 ms since the last sample) and non-raw / SDL paths.
    const bool raw_active = s_raw_owns.load(std::memory_order_acquire);
    bool push_now = !raw_active || sdl_dx != 0 || sdl_dy != 0 || s_hist_count == 0;
    if (!push_now) {
        const MouseSample& newest = s_hist[(s_hist_head + kMouseHist - 1) % kMouseHist];
        if (ns > newest.ns && (ns - newest.ns) > 12000000ull) push_now = true;
    }
    if (push_now) push_sample_locked(ns);
}

bool take_buttons(bool relative, uint32_t& buttons) {
#if defined(_WIN32)
    if (!relative) {
        s_raw_buttons.store(0, std::memory_order_relaxed);
        s_raw_pressed.store(0, std::memory_order_relaxed);
        return false;
    }
    if (!s_raw_owns.load(std::memory_order_acquire)) return false;
    if (!raw_counting()) {
        s_raw_buttons.store(0, std::memory_order_relaxed);
        s_raw_pressed.store(0, std::memory_order_relaxed);
        buttons = 0;
        return true;
    }
    const uint32_t latched = s_raw_pressed.exchange(0, std::memory_order_acq_rel);
    const uint32_t held = s_raw_buttons.load(std::memory_order_acquire);
    buttons = held | latched;
    return true;
#else
    (void)relative;
    (void)buttons;
    return false;
#endif
}

void note_scale(float degrees_per_count) {
    s_degrees_per_count.store(degrees_per_count, std::memory_order_relaxed);
}

ViewTag frame_tag(uint8_t* mem) {
    ViewTag tag;
    {
        std::lock_guard lock(s_count_lock);
        tag.taken_x = s_taken_x;
        tag.taken_y = s_taken_y;
        tag.taken_ns = s_taken_ns;
    }
    tag.degrees_per_count = s_degrees_per_count.load(std::memory_order_relaxed);
    if (!mem) return tag;
    std::lock_guard lock(s_lock);
    const uint32_t self = s_camera.self;
    // The camera was updated after the last VPADRead (this frame), by the
    // player, with the mouse path on and the mouse captured.
    if (!enabled() || !self || !s_camera_seen || s_last_update_frame != s_vpad_frames || s_debug.locked ||
        !(s_debug.weight > 0.0f) || !s_relative.load(std::memory_order_relaxed) ||
        s_last_blocked || tag.degrees_per_count == 0.0f || !valid_camera(mem, self))
        return tag;
    for (int i = 0; i < 3; ++i) {
        tag.at[i] = rf(mem, self + 0x48 + 4 * i);
        tag.eye[i] = rf(mem, self + 0x58 + 4 * i);
        if (!std::isfinite(tag.at[i]) || !std::isfinite(tag.eye[i])) return tag;
    }
    tag.pitch = s_debug.pitch;
    tag.pitch_min = s_debug.pitch_min;
    tag.pitch_max = s_debug.pitch_max;
    tag.active = true;
    return tag;
}

bool counts_now(int64_t& x, int64_t& y) {
    std::lock_guard lock(s_count_lock);
    x = s_taken_x + (s_raw_x - s_raw_taken_x);
    y = s_taken_y + (s_raw_y - s_raw_taken_y);
    return true;
}

bool counts_at(uint64_t query_ns, double& x, double& y) {
    std::lock_guard lock(s_count_lock);
    if (s_hist_count == 0) {
        x = double(s_taken_x + (s_raw_x - s_raw_taken_x));
        y = double(s_taken_y + (s_raw_y - s_raw_taken_y));
        return true;
    }
    const size_t oldest_idx = (s_hist_head + kMouseHist - s_hist_count) % kMouseHist;
    const auto at = [&](size_t i) -> const MouseSample& { return s_hist[(oldest_idx + i) % kMouseHist]; };
    const MouseSample& newest = at(s_hist_count - 1);
    if (query_ns >= newest.ns || s_hist_count == 1) {
        x = double(newest.x);
        y = double(newest.y);
        return true;
    }
    const MouseSample& oldest = at(0);
    if (query_ns <= oldest.ns) {
        x = double(oldest.x);
        y = double(oldest.y);
        return true;
    }
    size_t lo = 0, hi = s_hist_count - 1;
    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        if (at(mid).ns <= query_ns)
            lo = mid;
        else
            hi = mid;
    }
    const MouseSample& a = at(lo);
    const MouseSample& b = at(hi);
    const double span = double(b.ns - a.ns);
    const double u = span > 0.0 ? double(query_ns - a.ns) / span : 1.0;
    x = double(a.x) + u * double(b.x - a.x);
    y = double(a.y) + u * double(b.y - a.y);
    return true;
}
} // namespace rbrew_aim
