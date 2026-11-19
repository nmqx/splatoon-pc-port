// mouse_aim.h - PC mouse aiming: the mouse turns the game's player camera
// itself, one to one, instead of going through the right stick or the GamePad
// motion sensors (see mouse_aim.cpp).
#pragma once
#include <cstdint>

struct CPUState;

namespace rbrew_aim {
// Start-up, before guest code runs: binds the camera thunk the manifest's
// `[patches]` entry calls (0x025DB0D8 = bl 0x02D0F000).
void install(CPUState* cpu);

// The "mouse camera" path is selected (settings.ini mouse_camera,
// RBREW_MOUSE_CAMERA) together with direct mouse aiming.
bool enabled();

// Game thread, once per VPADRead. The mouse movement of this frame in camera
// degrees (yaw > 0 turns left, pitch > 0 looks up); stick_pitch is the right
// stick's vertical deflection. Returns true when the player camera takes the
// movement: the caller must then not give it to the virtual GamePad. False:
// no player camera is being updated (or the path is off) and the caller keeps
// its old path. `blocked` (ESC menu, fly camera) drops the movement.
bool vpad_frame(float yaw_deg, float pitch_deg, float stick_pitch, bool blocked);

// Pitch (degrees) the virtual GamePad rests at without a gyro: the attitude
// the game's motion aiming takes as "level" (GyroPitchBase, -60 degrees from
// the pad's forward axis).
constexpr float kPadRestPitch = -30.0f;

// What the camera thunk did last (diagnostics, RBREW_MOUSE_LOG and the test).
struct Debug {
    uint32_t camera = 0;       // guest object
    uint64_t updates = 0;      // camera updates seen
    bool locked = false;       // the game holds the camera (no player control)
    float yaw = 0, pitch = 0;  // degrees: the camera's direction after the update
    float game_pitch = 0;      // degrees: the game's own pitch (stick / motion)
    float offset = 0;          // degrees: what the mouse adds to it
    float pitch_max = 0, pitch_mid = 0, pitch_min = 0;
    float applied_yaw = 0, applied_pitch = 0; // degrees of mouse movement used by the last update
    uint64_t vpad_frame = 0;   // VPADRead count at the last update
    float weight = 0;          // the camera's input weight (0: the player has no control)
};
Debug debug();

// ---- Mouse look at the display's rate (gx2_frame_interp.cpp) ---------------
// The game turns the camera once per step (60 a second); the in-between
// frames the renderer draws (120/165/... FPS) add the mouse movement received
// since. For that the mouse is read at its own rate: a raw input thread
// (Windows) counts its movement as it comes, and VPADRead takes the counts
// from it (SDL only sees the mouse when the game thread pumps events, once
// per step).
//
// vpad.cpp, once per VPADRead: the mouse counts this frame gives the game.
// sdl_dx/dy: what SDL reported (or the autotest); relative: SDL's relative
// mode is on (the mouse is captured by the game window). dx/dy: the counts to
// use, SDL's plus the raw thread's since the last call.
void take_counts(int sdl_dx, int sdl_dy, bool relative, int& dx, int& dy);
// vpad.cpp, once per VPADRead (after take_counts): when the raw input thread
// owns mouse raw input on Windows, SDL2 ignores legacy WM_*BUTTON* messages in
// relative mode and stops updating SDL_GetMouseState. Returns true and writes
// the SDL_BUTTON(1..5) bitmask (held | pressed since last call) when the raw
// thread owns mouse input; false means fall back to SDL_GetMouseState.
bool take_buttons(bool relative, uint32_t& buttons);
// vpad.cpp, direct mouse aiming: camera degrees per count (sensitivity).
void note_scale(float degrees_per_count);
// A game frame's camera, taken at its swap (game thread, GX2SwapScanBuffers):
// the pose the frame was drawn with and how many mouse counts it contains.
struct ViewTag {
    bool active = false;              // the mouse turns the player camera (path on, camera in control)
    int64_t taken_x = 0, taken_y = 0; // mouse counts given to the game up to this frame's camera update
    uint64_t taken_ns = 0;            // steady-clock ns when taken_x/taken_y were latched
    float degrees_per_count = 0.0f;   // camera degrees per count: yaw = -x * it, pitch = -y * it
    float eye[3]{}, at[3]{};          // the frame's camera pose (world)
    float pitch = 0.0f, pitch_min = 0.0f, pitch_max = 0.0f; // degrees
};
ViewTag frame_tag(uint8_t* mem);
// Render thread: every mouse count received so far, given to the game or
// not (same units as ViewTag::taken_*). False: no player camera to turn.
bool counts_now(int64_t& x, int64_t& y);
// Render thread: interpolated mouse counts at steady-clock timestamp query_ns
// from the 1000 Hz raw-input history ring (clamped to newest/oldest sample,
// never extrapolated).
bool counts_at(uint64_t query_ns, double& x, double& y);

// RBREW_AIM_AUTOTEST=<log file> (development): the port plays the acceptance
// test by itself - it passes the title and the news broadcast with button
// presses, waits for the player camera, then plays mouse and stick traces and
// logs what the camera did. Real devices are ignored meanwhile.
struct TestPad {
    uint32_t buttons = 0;
    float lx = 0, ly = 0, rx = 0, ry = 0;
    int mdx = 0, mdy = 0;
};
bool autotest_frame(uint8_t* mem, TestPad& pad);
} // namespace rbrew_aim
