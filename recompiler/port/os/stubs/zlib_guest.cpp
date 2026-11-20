// zlib_guest.cpp - the game's zlib125 imports (inflate/deflate) on the host zlib.
//
// The v16 game never called them (blank stubs), but the v272 update's PIA 4.9.4 deflates its
// session data when a match starts: with the blank stub deflateInit2_ "failed" and the
// matchmaking ended in "A communications error occurred".
//
// Guest z_stream (zlib 1.2.5, 32-bit big-endian, 0x38 bytes):
//   +00 next_in  +04 avail_in  +08 total_in  +0C next_out  +10 avail_out  +14 total_out
//   +18 msg      +1C state     +20 zalloc    +24 zfree     +28 opaque     +2C data_type
//   +30 adler    +34 reserved
// Each guest stream gets a host z_stream; the guest's state pointer is set to the stream's own
// address (non-null, as the game may check it). Pointers are guest addresses translated with MEM.
#include "../os_common.h"
#include "stubs_addrs.h"
#ifdef RBREW_ZLIB_HEADER
#include RBREW_ZLIB_HEADER // static zlib, Z_PREFIX (CMakeLists.txt): no libz.dll
#else
#include <zlib.h>
#endif
#include <mutex>
#include <unordered_map>

namespace {
std::mutex s_mutex;
std::unordered_map<uint32_t, z_stream*> s_streams;

constexpr int kZStreamError = -2, kZMemError = -4;

z_stream* host_stream(uint32_t guest) {
    auto it = s_streams.find(guest);
    return it == s_streams.end() ? nullptr : it->second;
}

// guest -> host before a call
void load(CPUState* cpu, uint32_t g, z_stream* z) {
    const uint32_t next_in = rbrew_read32(MEM, g + 0x00), next_out = rbrew_read32(MEM, g + 0x0C);
    z->next_in = next_in ? MEM + next_in : Z_NULL;
    z->avail_in = rbrew_read32(MEM, g + 0x04);
    z->next_out = next_out ? MEM + next_out : Z_NULL;
    z->avail_out = rbrew_read32(MEM, g + 0x10);
}
// host -> guest after a call
void store(CPUState* cpu, uint32_t g, z_stream* z) {
    rbrew_write32(MEM, g + 0x00, z->next_in ? uint32_t(z->next_in - MEM) : 0u);
    rbrew_write32(MEM, g + 0x04, z->avail_in);
    rbrew_write32(MEM, g + 0x08, uint32_t(z->total_in));
    rbrew_write32(MEM, g + 0x0C, z->next_out ? uint32_t(z->next_out - MEM) : 0u);
    rbrew_write32(MEM, g + 0x10, z->avail_out);
    rbrew_write32(MEM, g + 0x14, uint32_t(z->total_out));
    rbrew_write32(MEM, g + 0x18, 0u); // msg
    rbrew_write32(MEM, g + 0x2C, uint32_t(z->data_type));
    rbrew_write32(MEM, g + 0x30, uint32_t(z->adler));
}

bool valid(CPUState* cpu, uint32_t g) {
    return g && rbrew_memory_accessible(MEM, g, 0x38);
}

z_stream* create(CPUState* cpu, uint32_t g) {
    std::lock_guard lock(s_mutex);
    if (z_stream* old = host_stream(g)) { // re-initialised without End: drop the old one
        inflateEnd(old);
        delete old;
        s_streams.erase(g);
    }
    auto* z = new z_stream{};
    s_streams[g] = z;
    rbrew_write32(MEM, g + 0x1C, g); // state: non-null
    return z;
}

void destroy(uint32_t g, bool deflating) {
    std::lock_guard lock(s_mutex);
    if (z_stream* z = host_stream(g)) {
        if (deflating) deflateEnd(z); else inflateEnd(z);
        delete z;
        s_streams.erase(g);
    }
}

z_stream* find(uint32_t g) {
    std::lock_guard lock(s_mutex);
    return host_stream(g);
}

// deflateInit2_(strm, level, method, windowBits, memLevel, strategy, version, stream_size)
void guest_deflateInit2_(CPUState* cpu) {
    const uint32_t g = ARG0;
    if (!valid(cpu, g)) { RET = uint32_t(kZStreamError); return; }
    z_stream* z = create(cpu, g);
    load(cpu, g, z);
    const int r = deflateInit2(z, int32_t(ARG1), int32_t(ARG2), int32_t(ARG3), int32_t(ARG4), int32_t(ARG5));
    store(cpu, g, z);
    if (r != Z_OK) destroy(g, true), rbrew_write32(MEM, g + 0x1C, 0u);
    RET = uint32_t(r);
}
void guest_deflate(CPUState* cpu) {
    const uint32_t g = ARG0;
    z_stream* z = valid(cpu, g) ? find(g) : nullptr;
    if (!z) { RET = uint32_t(kZStreamError); return; }
    load(cpu, g, z);
    const int r = deflate(z, int32_t(ARG1));
    store(cpu, g, z);
    RET = uint32_t(r);
}
void guest_deflateEnd(CPUState* cpu) {
    const uint32_t g = ARG0;
    if (!valid(cpu, g)) { RET = uint32_t(kZStreamError); return; }
    destroy(g, true);
    rbrew_write32(MEM, g + 0x1C, 0u);
    RET = Z_OK;
}
// deflateBound(strm, sourceLen)
void guest_deflateBound(CPUState* cpu) {
    const uint32_t g = ARG0;
    z_stream* z = valid(cpu, g) ? find(g) : nullptr;
    RET = uint32_t(deflateBound(z, uLong(ARG1)));
}
// inflateInit_(strm, version, stream_size)
void guest_inflateInit_(CPUState* cpu) {
    const uint32_t g = ARG0;
    if (!valid(cpu, g)) { RET = uint32_t(kZStreamError); return; }
    z_stream* z = create(cpu, g);
    load(cpu, g, z);
    const int r = inflateInit(z);
    store(cpu, g, z);
    if (r != Z_OK) destroy(g, false), rbrew_write32(MEM, g + 0x1C, 0u);
    RET = uint32_t(r);
}
// inflateInit2_(strm, windowBits, version, stream_size)
void guest_inflateInit2_(CPUState* cpu) {
    const uint32_t g = ARG0;
    if (!valid(cpu, g)) { RET = uint32_t(kZStreamError); return; }
    z_stream* z = create(cpu, g);
    load(cpu, g, z);
    const int r = inflateInit2(z, int32_t(ARG1));
    store(cpu, g, z);
    if (r != Z_OK) destroy(g, false), rbrew_write32(MEM, g + 0x1C, 0u);
    RET = uint32_t(r);
}
void guest_inflate(CPUState* cpu) {
    const uint32_t g = ARG0;
    z_stream* z = valid(cpu, g) ? find(g) : nullptr;
    if (!z) { RET = uint32_t(kZStreamError); return; }
    load(cpu, g, z);
    const int r = inflate(z, int32_t(ARG1));
    store(cpu, g, z);
    RET = uint32_t(r);
}
void guest_inflateEnd(CPUState* cpu) {
    const uint32_t g = ARG0;
    if (!valid(cpu, g)) { RET = uint32_t(kZStreamError); return; }
    destroy(g, false);
    rbrew_write32(MEM, g + 0x1C, 0u);
    RET = Z_OK;
}
} // namespace

void zlib_guest_register(CPUState* cpu) {
#ifdef ADDR_deflateInit2_
    rbrew_register_func(cpu, ADDR_deflateInit2_, guest_deflateInit2_);
#endif
#ifdef ADDR_deflate
    rbrew_register_func(cpu, ADDR_deflate, guest_deflate);
#endif
#ifdef ADDR_deflateEnd
    rbrew_register_func(cpu, ADDR_deflateEnd, guest_deflateEnd);
#endif
#ifdef ADDR_deflateBound
    rbrew_register_func(cpu, ADDR_deflateBound, guest_deflateBound);
#endif
#ifdef ADDR_inflateInit_
    rbrew_register_func(cpu, ADDR_inflateInit_, guest_inflateInit_);
#endif
#ifdef ADDR_inflateInit2_
    rbrew_register_func(cpu, ADDR_inflateInit2_, guest_inflateInit2_);
#endif
#ifdef ADDR_inflate
    rbrew_register_func(cpu, ADDR_inflate, guest_inflate);
#endif
#ifdef ADDR_inflateEnd
    rbrew_register_func(cpu, ADDR_inflateEnd, guest_inflateEnd);
#endif
    (void)kZMemError;
}
