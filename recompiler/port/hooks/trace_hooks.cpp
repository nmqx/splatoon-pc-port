// trace_hooks.cpp - development trace of guest function calls (opt-in).
//
// RBREW_TRACE_FUNCS=028C2218,028C63A4 logs each call of these functions (caller, r3..r6 on entry,
// r3 on return) to RBREW_TRACE_FILE (default trace_funcs.txt in the game folder), at most 4000
// lines. Each address must be listed in the manifest `[hooks]` (the recompiler only wraps those);
// an address without a wrapper is reported once and ignored.
#include "../runtime/rebrewu_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {
std::FILE* s_file = nullptr;
std::mutex s_mutex;
unsigned s_lines = 0;

void trace(CPUState* cpu, uint32_t address, HostFunc original) {
    const uint32_t lr = cpu->lr, r3 = cpu->r[3], r4 = cpu->r[4], r5 = cpu->r[5], r6 = cpu->r[6];
    original(cpu);
    std::lock_guard lock(s_mutex);
    if (!s_file || s_lines >= 4000) return;
    ++s_lines;
    std::fprintf(s_file, "%08X lr=%08X r3=%08X r4=%08X r5=%08X r6=%08X -> %08X\n", address, lr, r3, r4, r5, r6,
                 cpu->r[3]);
    std::fflush(s_file);
}
} // namespace

void trace_hooks_register() {
    const char* list = std::getenv("RBREW_TRACE_FUNCS");
    if (!list || !*list) return;
    const char* path = std::getenv("RBREW_TRACE_FILE");
    s_file = std::fopen(path && *path ? path : "trace_funcs.txt", "w");
    if (!s_file) return;
    std::string text(list);
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) end = text.size();
        const uint32_t address = uint32_t(std::strtoul(text.substr(start, end - start).c_str(), nullptr, 16));
        if (address) {
            const bool ok = rbrew_set_guest_hook(address, trace);
            std::fprintf(s_file, "# trace %08X %s\n", address, ok ? "armed" : "FAILED");
        }
        start = end + 1;
    }
    std::fflush(s_file);
}
