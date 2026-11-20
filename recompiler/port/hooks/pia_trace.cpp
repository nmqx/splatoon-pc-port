// pia_trace.cpp - PIA's own trace messages (v272 development aid).
//
// PIA calls a printf-like trace function (v272 02C3350C: trace object, flags, format, ...) at
// every step of its jobs ("NatDetecter::...", "JoinMeshJob::..."). The release game ships it
// empty. The manifest patches its first instruction into a branch to RBREW_GAME_PIA_TRACE_THUNK,
// bound here: with RBREW_PIA_TRACE=<file> each message is formatted and written to that file,
// otherwise the thunk returns at once (same as the original).
#include "../runtime/rebrewu_runtime.h"
#include "../runtime/game_revision.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#ifdef RBREW_GAME_HAS_PIA_TRACE_THUNK
namespace {
std::FILE* s_file = nullptr;
std::mutex s_mutex;
unsigned s_lines = 0;
const auto s_start = std::chrono::steady_clock::now();

std::string guest_string(CPUState* cpu, uint32_t a) {
    std::string out;
    for (uint32_t i = 0; a && i < 200 && rbrew_memory_accessible(cpu->mem, a + i, 1); ++i) {
        const char c = char(cpu->mem[a + i]);
        if (!c) break;
        out += c;
    }
    return out;
}

// PPC32 SysV varargs after (r3 object, r4 flags, r5 format): integers in r6..r10 then the
// caller's parameter save area (r1 + 8), 64-bit integers in an odd/even register pair;
// doubles in f1..f8.
struct Args {
    CPUState* cpu;
    unsigned gpr = 6, fpr = 1, stack = 0;
    uint32_t next32() {
        if (gpr <= 10) return cpu->r[gpr++];
        const uint32_t v = rbrew_read32(cpu->mem, cpu->r[1] + 8 + 4 * stack);
        ++stack;
        return v;
    }
    uint64_t next64() {
        if (gpr <= 9) {
            if (!(gpr & 1)) ++gpr; // pairs start on an odd register (r7:r8, r9:r10)
            const uint64_t v = (uint64_t(cpu->r[gpr]) << 32) | cpu->r[gpr + 1];
            gpr += 2;
            return v;
        }
        gpr = 11;
        if (stack & 1) ++stack;
        const uint64_t hi = next32(), lo = next32();
        return (hi << 32) | lo;
    }
    double nextd() { return fpr <= 8 ? cpu->f[fpr++] : 0.0; }
};

std::string format(CPUState* cpu, const std::string& fmt) {
    Args args{cpu};
    std::string out;
    for (size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') {
            out += fmt[i];
            continue;
        }
        size_t j = i + 1;
        std::string spec = "%";
        while (j < fmt.size() && std::strchr("-+ #0123456789.", fmt[j])) spec += fmt[j++];
        int longs = 0;
        while (j < fmt.size() && (fmt[j] == 'l' || fmt[j] == 'h' || fmt[j] == 'z')) longs += fmt[j++] == 'l';
        if (j >= fmt.size()) break;
        const char c = fmt[j];
        char buf[256];
        buf[0] = 0;
        switch (c) {
        case 'd': case 'i':
            if (longs >= 2) std::snprintf(buf, sizeof buf, (spec + "lld").c_str(), (long long)args.next64());
            else std::snprintf(buf, sizeof buf, (spec + "d").c_str(), int32_t(args.next32()));
            break;
        case 'u': case 'x': case 'X': case 'o':
            if (longs >= 2) std::snprintf(buf, sizeof buf, (spec + "ll" + c).c_str(), (unsigned long long)args.next64());
            else std::snprintf(buf, sizeof buf, (spec + c).c_str(), args.next32());
            break;
        case 'p': std::snprintf(buf, sizeof buf, "%08X", args.next32()); break;
        case 'c': std::snprintf(buf, sizeof buf, "%c", char(args.next32())); break;
        case 's': out += guest_string(cpu, args.next32()); break;
        case 'f': case 'g': case 'e': std::snprintf(buf, sizeof buf, (spec + c).c_str(), args.nextd()); break;
        case '%': out += '%'; break;
        default: out += spec + c; break;
        }
        out += buf;
        i = j;
    }
    return out;
}

void pia_trace(CPUState* cpu) {
    if (!s_file) return;
    const std::string text = format(cpu, guest_string(cpu, cpu->r[5]));
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s_start).count();
    std::lock_guard lock(s_mutex);
    if (s_lines >= 20000) return;
    ++s_lines;
    std::fprintf(s_file, "%10.1f lr=%08X %s\n", ms, cpu->lr, text.c_str());
    std::fflush(s_file);
}
} // namespace

void pia_trace_install(CPUState* cpu) {
    if (const char* path = std::getenv("RBREW_PIA_TRACE"); path && *path) s_file = std::fopen(path, "w");
    rbrew_register_func(cpu, RBREW_GAME_PIA_TRACE_THUNK, pia_trace);
}
#else
void pia_trace_install(CPUState*) {}
#endif
