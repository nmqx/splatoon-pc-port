// broadcast_guard.cpp - message broadcast with a loop guard (Splatoon v288).
//
// Touching the GamePad screen froze the game: its main thread never left
// 0285C3D0, the game's message broadcast (a group list of receivers, each with
// a list of handlers called through a pointer-to-member). A memory snapshot showed sane
// lists at that moment, so the walk does not end because the lists change while it runs.
//
// The recompiler wraps 0285C3D0 (`[hooks]` of the manifest) and this hook runs
// the same walk natively, step for step with the guest code:
//   0285BD54(obj, -1); group = [key]; head H = obj+0x34+16*group, node offset [H+0xC];
//   for each receiver e: [obj+0x218] = 0; its handler list H2 = [[[obj+0x30]+4*[e+8]]+8];
//   for each handler x: [obj+0x218] = ++n; if [key] == 5 or [x+0x1C] == 2, call
//   the member (msg: s16 this delta, s16 vtable index (<0: plain function), u32 function/offset);
//   then [obj+0x218] = -1; 0285BD54(obj, -1).
// It adds a step limit: past it the walk stops, the details go to stderr once
// per call site, and the game goes on (one message not delivered to everyone).
// RBREW_BROADCAST_GUARD=0 runs the original code instead.
#include "../runtime/rebrewu_runtime.h"
#include "../runtime/game_revision.h"
#include "../runtime/splatoon_diagnostics.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef RBREW_GAME_HAS_BROADCAST_DISPATCH // v272, v288
namespace {
constexpr uint32_t kBroadcast = RBREW_GAME_BROADCAST_DISPATCH;
constexpr uint32_t kSetCurrent = RBREW_GAME_BROADCAST_SET_CURRENT; // called with (obj, -1) before and after the walk
constexpr uint32_t kMaxSteps = 50000; // receivers + handler calls; real broadcasts are far below

uint32_t rd(CPUState* cpu, uint32_t a) {
    return rbrew_read32(cpu->mem, a);
}
void wr(CPUState* cpu, uint32_t a, uint32_t v) {
    rbrew_write32(cpu->mem, a, v);
}
int32_t rh(CPUState* cpu, uint32_t a) {
    return int16_t(rbrew_read16(cpu->mem, a));
}
bool ok(CPUState* cpu, uint32_t a, uint32_t n) {
    return a && uint64_t(a) + n <= cpu->guest_memory_size && rbrew_memory_accessible(cpu->mem, a, n);
}
bool node_linked(CPUState* cpu, uint32_t node) {
    return ok(cpu, node, 8) && rd(cpu, node) != 0 && rd(cpu, node + 4) != 0;
}
void call(CPUState* cpu, uint32_t fn) {
    rbrew_call(cpu, fn);
}

void report_unlink(const char* kind, uint32_t caller, uint32_t obj, uint32_t key_val, uint32_t list_head,
                   uint32_t elem, uint32_t pre_next, bool resumed) {
    static std::atomic<uint32_t> s_unlinks{0};
    const uint32_t count = s_unlinks.fetch_add(1, std::memory_order_relaxed);
    if (count >= 8) return;
    char diag[50];
    std::snprintf(diag, sizeof diag, "bcast unlink %s k=%u x=%08X %s", kind, key_val, elem, resumed ? "res" : "brk");
    SplatoonDiagnostics::RecordEvent(SplatoonDiagnostics::EventCategory::Engine,
                                     SplatoonDiagnostics::EventSeverity::Warning, 1, diag);
    RBREW_LOG_WARN("broadcast",
                   "unlinked %s node during dispatch (%s): caller lr=%08X obj=%08X key=%u head=%08X at=%08X "
                   "pre_next=%08X\n",
                   kind, resumed ? "resumed at pre_next" : "stopped walk", caller, obj, key_val, list_head, elem,
                   pre_next);
}

void report(CPUState* cpu, uint32_t caller, uint32_t obj, uint32_t key, uint32_t msg, uint32_t head, uint32_t inner,
            uint32_t offset, uint32_t x) {
    static std::atomic<uint32_t> s_reported{0};
    if (s_reported.fetch_add(1, std::memory_order_relaxed) >= 8) return;
    const uint32_t key_val = ok(cpu, key, 4) ? rd(cpu, key) : 0xFFFFFFFFu;
    char diag[50];
    std::snprintf(diag, sizeof diag, "bcast cap %u k=%u x=%08X", kMaxSteps, key_val, x);
    SplatoonDiagnostics::RecordEvent(SplatoonDiagnostics::EventCategory::Engine,
                                     SplatoonDiagnostics::EventSeverity::Warning, 2, diag);
    std::string handlers;
    uint32_t node = (inner && ok(cpu, inner, 8)) ? rd(cpu, inner + 4) : 0;
    for (int i = 0; i < 16 && node && node != inner && ok(cpu, node, 8); ++i) {
        const uint32_t h = node - offset;
        char item[64];
        std::snprintf(item, sizeof item, " %08X(vt %08X flag %u)", h, ok(cpu, h, 0x20) ? rd(cpu, h + 4) : 0,
                      ok(cpu, h, 0x20) ? rd(cpu, h + 0x1C) : 0);
        handlers += item;
        node = rd(cpu, node + 4);
    }
    RBREW_LOG_WARN("broadcast",
                   "loop stopped after %u steps: caller lr=%08X obj=%08X key=%u msg=%08X/%08X/%08X group head=%08X "
                   "handler head=%08X offset=%X at=%08X handlers:%s%s\n",
                   kMaxSteps, caller, obj, key_val, rd(cpu, msg), rd(cpu, msg + 4),
                   ok(cpu, msg, 12) ? rd(cpu, msg + 8) : 0, head, inner, offset, x, handlers.c_str(),
                   node == inner ? " (end)" : " ...");
}

void broadcast(CPUState* cpu, uint32_t, HostFunc original) {
    const uint32_t obj = cpu->r[3], key = cpu->r[4], msg = cpu->r[5], caller = cpu->lr;
    if (!ok(cpu, obj, 0x21C) || !ok(cpu, key, 4) || !ok(cpu, msg, 8) || rd(cpu, key) > 0x1000) {
        original(cpu);
        return;
    }
    const uint32_t head = obj + 0x34 + rd(cpu, key) * 16;
    if (!ok(cpu, head, 0x10)) {
        original(cpu);
        return;
    }
    // A frame of our own, like the guest's (stwu r1,-0x40): callees save LR at 4(r1).
    const uint32_t sp = cpu->r[1], lr = cpu->lr;
    cpu->r[1] = sp - 0x40;
    wr(cpu, cpu->r[1], sp);
    auto set_current = [&] {
        cpu->r[3] = obj;
        cpu->r[4] = 0xFFFFFFFFu;
        cpu->lr = kBroadcast + 0x38u;
        call(cpu, kSetCurrent);
    };
    set_current();
    const uint32_t offset = rd(cpu, head + 0xC);
    uint32_t e_node = rd(cpu, head + 4), steps = 0;
    while (e_node && e_node != head) {
        if (!ok(cpu, e_node, 8)) break;
        const uint32_t pre_e_next = rd(cpu, e_node + 4);
        const uint32_t e = e_node - offset;
        if (!ok(cpu, e, 0xC)) break;
        wr(cpu, obj + 0x218, 0);
        const uint32_t table = rd(cpu, obj + 0x30);
        const uint32_t slot_addr = table + 4 * rd(cpu, e + 8);
        if (ok(cpu, slot_addr, 4)) {
            const uint32_t list = rd(cpu, slot_addr);
            const uint32_t inner = ok(cpu, list, 0xC) ? rd(cpu, list + 8) : 0;
            if (ok(cpu, inner, 0x10)) {
                const uint32_t o2 = rd(cpu, inner + 0xC);
                uint32_t x_node = rd(cpu, inner + 4);
                if (x_node && x_node != inner) {
                    const int32_t delta = rh(cpu, msg), index = rh(cpu, msg + 2);
                    const uint32_t function = rd(cpu, msg + 4);
                    uint32_t n = 0;
                    for (;;) {
                        if (!ok(cpu, x_node, 8)) break;
                        const uint32_t pre_x_next = rd(cpu, x_node + 4);
                        const uint32_t x = x_node - o2;
                        if (!ok(cpu, x, 0x20)) break;
                        ++n;
                        wr(cpu, obj + 0x218, n);
                        if (rd(cpu, key) == 5 || rd(cpu, x + 0x1C) == 2) {
                            const uint32_t self = x + uint32_t(delta);
                            uint32_t target = 0, r5 = 0;
                            bool valid_target = true;
                            if (index < 0) {
                                target = function;
                                r5 = n - 1;
                            } else {
                                const uint32_t vt_ptr_addr = self + uint32_t(int32_t(int16_t(function)));
                                if (!ok(cpu, vt_ptr_addr, 4)) {
                                    valid_target = false;
                                } else {
                                    r5 = rd(cpu, vt_ptr_addr) + uint32_t(index) * 8;
                                    if (!ok(cpu, r5, 8)) {
                                        valid_target = false;
                                    } else {
                                        target = rd(cpu, r5 + 4);
                                    }
                                }
                            }
                            if (valid_target && target) {
                                cpu->r[3] = self;
                                cpu->r[4] = n;
                                cpu->r[5] = r5;
                                cpu->lr = kBroadcast + 0xE0u;
                                call(cpu, target);
                            }
                        }
                        if (++steps > kMaxSteps) {
                            report(cpu, caller, obj, key, msg, head, inner, o2, x);
                            goto done;
                        }
                        uint32_t next_x_node = ok(cpu, x_node, 8) ? rd(cpu, x_node + 4) : 0;
                        if (next_x_node == 0) {
                            // A handler unlinked itself during dispatch (sead::ListNode::erase
                            // clears prev/next to 0). Without this check, `0 - o2` wraps to
                            // 0xFFFFFFF0 and spins forever. Resume at pre_x_next if still linked.
                            const bool resumed = (pre_x_next == inner) || node_linked(cpu, pre_x_next);
                            report_unlink("inner", caller, obj, rd(cpu, key), inner, x, pre_x_next, resumed);
                            if (pre_x_next == inner || !resumed) break;
                            next_x_node = pre_x_next;
                        }
                        x_node = next_x_node;
                        if (x_node == inner) break;
                        n = rd(cpu, obj + 0x218);
                    }
                }
            }
        }
        if (++steps > kMaxSteps) {
            report(cpu, caller, obj, key, msg, head, 0, 0, e);
            break;
        }
        uint32_t next_e_node = ok(cpu, e_node, 8) ? rd(cpu, e_node + 4) : 0;
        if (next_e_node == 0) {
            const bool resumed = (pre_e_next == head) || node_linked(cpu, pre_e_next);
            report_unlink("outer", caller, obj, rd(cpu, key), head, e, pre_e_next, resumed);
            if (pre_e_next == head || !resumed) break;
            next_e_node = pre_e_next;
        }
        e_node = next_e_node;
    }
done:
    wr(cpu, obj + 0x218, 0xFFFFFFFFu);
    set_current();
    cpu->r[1] = sp;
    cpu->lr = lr;
}
} // namespace

void broadcast_guard_install() {
    if (const char* v = std::getenv("RBREW_BROADCAST_GUARD"); v && std::strcmp(v, "0") == 0) return;
    if (!rbrew_set_guest_hook(kBroadcast, broadcast)) RBREW_LOG_WARN("broadcast", "could not register the hook\n");
}
#else
void broadcast_guard_install() {}
#endif
