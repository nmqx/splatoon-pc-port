// gx2_frame_interp.h - real in-between frames: 120, 165, 180, 240 pictures a
// second or one per display refresh, from a game that steps 60 times.
//
// Splatoon steps its logic once per frame (player, physics and ink have no
// time step), so more real frames per second with normal game speed cannot
// come from running the game faster. Instead the game keeps 60 logic steps
// per second and the render thread draws the pictures between them:
//
//   1. Frame N runs normally. Each draw's uniform data (camera, object and
//      bone matrices, ...) is recorded, and its Vulkan commands are kept.
//   2. At N's present the finished image is copied aside, the game continues,
//      and the render thread replays N's commands once per in-between
//      picture, with each draw's transforms blended between N-1 and N (draws
//      matched by program, vertex buffer, count and occurrence) at the
//      position where that picture's present time falls. Every picture is
//      copied aside and presented at its time.
//
// The schedule: presents are evenly spaced, `rate` of them per game step
// (120: 2, 165: 2.75, 180: 3, 240: 4, display: refresh rate / 59.94). The
// position of the next picture runs on from step to step, so the rate need
// not be whole: at 165 a step gets two or three pictures, at positions on
// elevenths. On such a grid frame N itself is one of the pictures; at other
// rates every picture is an in-between frame. The display therefore shows
// the game up to one step late (120: half a step, 240: three quarters).
//
// A controller keeps the game's 60 steps: when the game slows down it moves
// to the next lower whole rate (still evenly spaced), down to none.
//
// The replay never writes guest memory (queries, stream-out, readbacks are
// skipped), so game logic only ever sees frame N. Menu "Images par seconde",
// settings.ini frame_rate, RBREW_FRAME_RATE=120|165|180|240|display.
#pragma once
#include <cstddef>
#include <cstdint>

struct Gx2AsyncWork;
namespace rbrew_aim {
struct ViewTag;
}

namespace gx2interp {
// Render thread, right before gx2_render_present of the frame it belongs to:
// that frame's camera and mouse counts (hooks/mouse_aim.h), for the mouse
// look at the display's rate in the in-between frames.
void set_view(const rbrew_aim::ViewTag& view);
// Game thread, per swap: interpolation wanted for the frames now being made.
void set_wanted(bool wanted);
bool mode_enabled(); // frame_rate above 60 (and no optical-flow frame gen)

// Render thread: what the schedule does, for the FPS counters and the menu.
struct Status {
    bool active = false;     // in-between frames are being presented
    float target_fps = 0.0f; // the chosen rate after the display's limit
    float level_fps = 0.0f;  // the rate the controller allows now
    float display_hz = 0.0f; // 0: unknown
    bool capped = false;     // the choice is above what the display shows
    bool vrr = false;        // G-Sync/FreeSync presents (the display follows each picture)
    bool locked = false;     // fixed refresh: one picture per refresh
};
Status status();
// The rate a menu choice would be presented at on this display.
Status describe(int frame_rate, bool gsync);
// Copies of the scan image that wait for their present time.
constexpr unsigned kSlots = 16;

// Render thread.
bool capturing();                                       // recording the current frame
bool replaying();                                       // running the in-between replay
void keep(Gx2AsyncWork* work);                          // a work that ran; takes ownership
void draw_unit(uint64_t key, bool transparent = false); // a draw (one instance-loop iteration) starts
bool replay_draw_known();                               // replay: this draw was recorded in frame N
void add_busy(uint64_t ns);                             // render-thread time spent on the frame's work
void record_uf(unsigned stage, const void* data, size_t size);
void record_bank(unsigned stage, unsigned bank, const void* data, size_t size);
// The uniform buffer (Vulkan handle, offset) a draw bound for stage/bank
// (255: the register block), for the Vulkan replay's substitution.
void note_ubo(unsigned stage, unsigned bank, uint64_t buffer, uint64_t offset);
void present_begin(); // gx2_render_present starts (not part of the frame)
void before_item();   // render thread: a queue item is about to run
// Replay: blended bytes for this draw, or false/nullptr (use guest memory).
bool replay_uf(unsigned stage, void* data, size_t size);
const uint8_t* replay_bank(unsigned stage, unsigned bank, size_t size);
// gx2_render_present: true when this frame's pictures are presented by the
// schedule (the caller must not present it now).
bool frame_complete(unsigned scan_index);
// Render thread, before the CPU overwrites a host-visible buffer (or recycles
// a descriptor pool) whose last recorded use has `last_serial` and whose GPU
// work is known complete: true while an in-between replay that re-issues that
// frame's commands may still read it on the GPU. The replay re-binds frame N's
// buffers without going through the caches that track their use, so their
// own serials say "free" too early: a vertex buffer rewritten in place for
// frame N+1 before the replay ran drew N's draw with N+1's vertices (the large
// flat gray triangle of the shooting range's HUD panel at 165 FPS, 2026-11-18).
// The caller then orphans (allocates anew) instead of writing in place.
bool replay_may_read(uint64_t last_serial);
// Render-thread loop: draws the pending in-between frames, presents the
// pictures whose time has come.
void service();
// Earliest time (steady clock ns) service() has work, or 0.
uint64_t deadline_ns();
// Render thread, between two draws of the frame being made: presents the
// pictures whose time has come (a present closes the rendering instance and
// the submission, like the renderer's own periodic submit).
void poll();
} // namespace gx2interp

// Implemented by the renderer (gx2_render_vk.cpp).
// Copies the scan image into slot (< gx2interp::kSlots). in_between: 0 for
// the game's own frame, else the number of the in-between frame within its
// step (frame dumps).
void gx2_render_interp_keep(unsigned scan_index, unsigned slot, unsigned in_between);
void gx2_render_interp_present(unsigned slot);
