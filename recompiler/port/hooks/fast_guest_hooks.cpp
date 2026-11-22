// fast_guest_hooks.cpp - Native C++ fast-paths for verified hot Splatoon v272 guest functions.
//
// Each function matches the exact PowerPC EABI register, memory, aliasing, and
// edge-case semantics of the generated v272 code while eliminating tight spin
// starvation and paired-single register/stack emulation overhead:
//   - 0x028F5B9C: multi-core worker job join / spin-wait barrier (progressive backoff)
//   - 0x02D1D730: nw::g3d::Sphere::Transform(out, in, mtx34)
//   - 0x02D1D830: nw::g3d::Sphere::Merge(out, a, b)
//   - 0x02D34540: PSMTX44Concat(A, B, AB)
//   - 0x02D349EC: PSMTXConcat(A, B, AB)
//   - 0x02D34AD0: PSMTXInverse(src, inv) -> r3 (1 = ok, 0 = singular)
//
// Set RBREW_FAST_GUEST_HOOKS=0 to disable all hooks for A/B comparison.
#include "fast_guest_hooks.h"
#include "../runtime/rebrewu_runtime.h"
#include "../runtime/game_revision.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <thread>
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

extern "C" bool g_rbrew_fast_guest_hooks = false;

namespace {

inline void spin_backoff(uint32_t& spin) {
    // Prevent the compiler from hoisting guest memory reads across iterations
    // and reduce SMT/cache-line contention while waiting on worker cores.
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (spin < 64u) {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#endif
    } else if (spin < 256u) {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
        _mm_pause();
        _mm_pause();
        _mm_pause();
#endif
    } else {
#if defined(_WIN32)
        SwitchToThread();
#else
        std::this_thread::yield();
#endif
    }
    ++spin;
}

inline float rd_f32(uint8_t* __restrict mem, uint32_t addr) {
    return static_cast<float>(rbrew_read_f32(mem, addr));
}

inline void wr_f32(uint8_t* __restrict mem, uint32_t addr, float v) {
    rbrew_write_f32(mem, addr, static_cast<double>(v));
}

inline uint32_t rd_u32_volatile(uint8_t* __restrict mem, uint32_t addr) {
    std::atomic_signal_fence(std::memory_order_acquire);
    return rbrew_read32(mem, addr);
}

#if RBREW_GAME_REVISION == 272
void hook_worker_join(CPUState* cpu, uint32_t, HostFunc) {
    rbrew_fast_v272_worker_join(cpu, cpu->mem);
}
void hook_sphere_transform(CPUState* cpu, uint32_t, HostFunc) {
    rbrew_fast_v272_sphere_transform(cpu, cpu->mem);
}
void hook_sphere_merge(CPUState* cpu, uint32_t, HostFunc) {
    rbrew_fast_v272_sphere_merge(cpu, cpu->mem);
}
void hook_mtx44_concat(CPUState* cpu, uint32_t, HostFunc) {
    rbrew_fast_v272_mtx44_concat(cpu, cpu->mem);
}
void hook_mtx_concat(CPUState* cpu, uint32_t, HostFunc) {
    rbrew_fast_v272_mtx_concat(cpu, cpu->mem);
}
void hook_mtx_inverse(CPUState* cpu, uint32_t, HostFunc) {
    rbrew_fast_v272_mtx_inverse(cpu, cpu->mem);
}
#endif

} // namespace

// 0x028F5B9C (v272): Multi-core worker job join / spin-wait barrier.
extern "C" void rbrew_fast_v272_worker_join(CPUState* __restrict cpu, uint8_t* __restrict mem) {
    const uint32_t self = cpu->r[3];
    const uint32_t saved_sp = cpu->r[1];
    const uint32_t saved_lr = cpu->lr;
    // Allocate matching 0x430-byte PPC stack frame so virtual callees saving LR at 4(r1) are safe.
    const uint32_t frame_sp = saved_sp - 0x430u;
    rbrew_write32(mem, frame_sp, saved_sp);
    cpu->r[1] = frame_sp;
    rbrew_write32(mem, frame_sp + 0x434u, saved_lr);

    if (rbrew_read8(mem, self + 0x54u) == 0u) {
        const uint32_t job_count = rbrew_read32(mem, self + 0x50u);
        for (uint32_t i = 0; i < job_count; ++i) {
            const uint32_t cap = rbrew_read32(mem, self + 0x48u);
            const uint32_t arr = rbrew_read32(mem, self + 0x4Cu);
            const uint32_t job_ptr = rbrew_read32(mem, arr + (i < cap ? (i << 2) : 0u));
            rbrew_write32(mem, self + 0x58u, job_ptr);

            if ((rbrew_read32(mem, job_ptr + 0x14u) & 1u) != 0u) {
                const uint32_t warr = rbrew_read32(mem, self + 0x44u);
                const uint32_t worker0 = rbrew_read32(mem, warr);
                const uint32_t vtbl = rbrew_read32(mem, worker0 + 0x0Cu);
                const uint32_t fn = rbrew_read32(mem, vtbl + 0x94u);
                cpu->r[3] = worker0;
                cpu->ctr = fn;
                cpu->lr = 0x028F5C64u;
                rbrew_call(cpu, fn);
            }

            uint32_t spin = 0;
            for (;;) {
                const uint32_t cap2 = rd_u32_volatile(mem, self + 0x48u);
                const uint32_t arr2 = rd_u32_volatile(mem, self + 0x4Cu);
                const uint32_t job = rd_u32_volatile(mem, arr2 + (i < cap2 ? (i << 2) : 0u));
                const uint32_t done_sum = rd_u32_volatile(mem, job + 0x30u) + rd_u32_volatile(mem, job + 0x34u) +
                                          rd_u32_volatile(mem, job + 0x38u);
                const uint32_t vtbl = rbrew_read32(mem, job);
                const uint32_t fn = rbrew_read32(mem, vtbl + 0x24u);
                cpu->r[3] = job;
                cpu->ctr = fn;
                cpu->lr = 0x028F5CF0u;
                rbrew_call(cpu, fn);
                if (done_sum >= cpu->r[3]) break;
                spin_backoff(spin);
            }
        }
        rbrew_write32(mem, self + 0x58u, 0u);
    }

    // Wait until every worker thread reports idle (worker+0x9C == 0).
    {
        uint32_t spin = 0;
        for (;;) {
            const int32_t nworkers = static_cast<int32_t>(rd_u32_volatile(mem, self + 0x40u));
            bool all_idle = true;
            if (nworkers > 0) {
                const uint32_t warr = rd_u32_volatile(mem, self + 0x44u);
                for (int32_t w = 0; w < nworkers; ++w) {
                    const uint32_t worker = rd_u32_volatile(mem, warr + (static_cast<uint32_t>(w) << 2));
                    if (rd_u32_volatile(mem, worker + 0x9Cu) != 0u) {
                        all_idle = false;
                        break;
                    }
                }
            }
            if (all_idle) break;
            spin_backoff(spin);
        }
    }

    cpu->r[3] = self;
    cpu->lr = 0x028F5D84u;
    rbrew_call(cpu, 0x028F5B34u);
    if (cpu->r[3] == 0u) {
        cpu->r[3] = frame_sp + 0x10u;
        cpu->r[4] = 0x100u;
        cpu->r[5] = 4u;
        cpu->r[6] = 0x028F5F58u;
        cpu->lr = 0x028F5DA4u;
        rbrew_call(cpu, 0x02D4E2A8u);
    }

    rbrew_write8(mem, self + 0x68u, 0u);
    rbrew_write32(mem, self + 0x50u, 0u);
    cpu->r[1] = saved_sp;
    cpu->lr = saved_lr;
}

// 0x02D1D730 (v272): nw::g3d::Sphere::Transform(out=r3, in=r4, mtx34=r5).
extern "C" void rbrew_fast_v272_sphere_transform(CPUState* __restrict cpu, uint8_t* __restrict mem) {
    const uint32_t out = cpu->r[3];
    const uint32_t in = cpu->r[4];
    const uint32_t mtx = cpu->r[5];

    const float cx = rd_f32(mem, in + 0u);
    const float cy = rd_f32(mem, in + 4u);
    const float cz = rd_f32(mem, in + 8u);
    const float r = rd_f32(mem, in + 12u);

    const float m00 = rd_f32(mem, mtx + 0u), m01 = rd_f32(mem, mtx + 4u), m02 = rd_f32(mem, mtx + 8u),
                m03 = rd_f32(mem, mtx + 12u);
    const float m10 = rd_f32(mem, mtx + 16u), m11 = rd_f32(mem, mtx + 20u), m12 = rd_f32(mem, mtx + 24u),
                m13 = rd_f32(mem, mtx + 28u);
    const float m20 = rd_f32(mem, mtx + 32u), m21 = rd_f32(mem, mtx + 36u), m22 = rd_f32(mem, mtx + 40u),
                m23 = rd_f32(mem, mtx + 44u);

    const float ox = m00 * cx + m01 * cy + m02 * cz + m03;
    const float oy = m10 * cx + m11 * cy + m12 * cz + m13;
    const float oz = m20 * cx + m21 * cy + m22 * cz + m23;

    const float s0 = m00 * m00 + m10 * m10 + m20 * m20;
    const float s1 = m01 * m01 + m11 * m11 + m21 * m21;
    const float s2 = m02 * m02 + m12 * m12 + m22 * m22;
    const float s_max = std::max(std::max(s0, s1), s2);
    const float scale = (s_max == 0.0f) ? 0.0f : std::sqrt(s_max);

    wr_f32(mem, out + 0u, ox);
    wr_f32(mem, out + 4u, oy);
    wr_f32(mem, out + 8u, oz);
    wr_f32(mem, out + 12u, r * scale);
}

// 0x02D1D830 (v272): nw::g3d::Sphere::Merge(out=r3, a=r4, b=r5).
extern "C" void rbrew_fast_v272_sphere_merge(CPUState* __restrict cpu, uint8_t* __restrict mem) {
    const uint32_t out = cpu->r[3];
    const uint32_t a = cpu->r[4];
    const uint32_t b = cpu->r[5];

    const float ax = rd_f32(mem, a + 0u), ay = rd_f32(mem, a + 4u), az = rd_f32(mem, a + 8u), ar = rd_f32(mem, a + 12u);
    const float bx = rd_f32(mem, b + 0u), by = rd_f32(mem, b + 4u), bz = rd_f32(mem, b + 8u), br = rd_f32(mem, b + 12u);

    const float dx = bx - ax;
    const float dy = by - ay;
    const float dz = bz - az;
    const float dr = br - ar;
    const float dr2 = dr * dr;
    constexpr float kEps = 1e-4f; // 0x38D1B717 @ 0x101CAD08
    const float dr2_eff = (dr2 >= kEps) ? dr2 : kEps;
    const float dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 < dr2_eff) {
        const uint32_t src = (ar > br) ? a : b;
        if (out != src) {
            const uint32_t w0 = rbrew_read32(mem, src + 0u);
            const uint32_t w1 = rbrew_read32(mem, src + 4u);
            const uint32_t w2 = rbrew_read32(mem, src + 8u);
            const uint32_t w3 = rbrew_read32(mem, src + 12u);
            rbrew_write32(mem, out + 0u, w0);
            rbrew_write32(mem, out + 4u, w1);
            rbrew_write32(mem, out + 8u, w2);
            rbrew_write32(mem, out + 12u, w3);
        }
        return;
    }

    const float dist = std::sqrt(dist2);
    const float new_r = 0.5f * (ar + br + dist);
    const float t = (new_r - ar) / dist;
    wr_f32(mem, out + 0u, ax + dx * t);
    wr_f32(mem, out + 4u, ay + dy * t);
    wr_f32(mem, out + 8u, az + dz * t);
    wr_f32(mem, out + 12u, new_r);
}

// 0x02D34540 (v272): PSMTX44Concat(A=r3, B=r4, AB=r5).
extern "C" void rbrew_fast_v272_mtx44_concat(CPUState* __restrict cpu, uint8_t* __restrict mem) {
    const uint32_t pa = cpu->r[3];
    const uint32_t pb = cpu->r[4];
    const uint32_t po = cpu->r[5];

    float a[16], b[16], o[16];
    for (uint32_t i = 0; i < 16; ++i) a[i] = rd_f32(mem, pa + (i << 2));
    for (uint32_t i = 0; i < 16; ++i) b[i] = rd_f32(mem, pb + (i << 2));

    for (uint32_t r = 0; r < 4; ++r) {
        const float a0 = a[r * 4 + 0], a1 = a[r * 4 + 1], a2 = a[r * 4 + 2], a3 = a[r * 4 + 3];
        o[r * 4 + 0] = a0 * b[0] + a1 * b[4] + a2 * b[8] + a3 * b[12];
        o[r * 4 + 1] = a0 * b[1] + a1 * b[5] + a2 * b[9] + a3 * b[13];
        o[r * 4 + 2] = a0 * b[2] + a1 * b[6] + a2 * b[10] + a3 * b[14];
        o[r * 4 + 3] = a0 * b[3] + a1 * b[7] + a2 * b[11] + a3 * b[15];
    }

    for (uint32_t i = 0; i < 16; ++i) wr_f32(mem, po + (i << 2), o[i]);
}

// 0x02D349EC (v272): PSMTXConcat(A=r3, B=r4, AB=r5) for 3x4 affine matrices.
extern "C" void rbrew_fast_v272_mtx_concat(CPUState* __restrict cpu, uint8_t* __restrict mem) {
    const uint32_t pa = cpu->r[3];
    const uint32_t pb = cpu->r[4];
    const uint32_t po = cpu->r[5];

    float a[12], b[12], o[12];
    for (uint32_t i = 0; i < 12; ++i) a[i] = rd_f32(mem, pa + (i << 2));
    for (uint32_t i = 0; i < 12; ++i) b[i] = rd_f32(mem, pb + (i << 2));

    for (uint32_t r = 0; r < 3; ++r) {
        const float a0 = a[r * 4 + 0], a1 = a[r * 4 + 1], a2 = a[r * 4 + 2], a3 = a[r * 4 + 3];
        o[r * 4 + 0] = a0 * b[0] + a1 * b[4] + a2 * b[8];
        o[r * 4 + 1] = a0 * b[1] + a1 * b[5] + a2 * b[9];
        o[r * 4 + 2] = a0 * b[2] + a1 * b[6] + a2 * b[10];
        o[r * 4 + 3] = a0 * b[3] + a1 * b[7] + a2 * b[11] + a3;
    }

    for (uint32_t i = 0; i < 12; ++i) wr_f32(mem, po + (i << 2), o[i]);
}

// 0x02D34AD0 (v272): PSMTXInverse(src=r3, inv=r4) -> r3 (1 = inverted, 0 = singular).
extern "C" void rbrew_fast_v272_mtx_inverse(CPUState* __restrict cpu, uint8_t* __restrict mem) {
    const uint32_t src = cpu->r[3];
    const uint32_t inv = cpu->r[4];

    const float m00 = rd_f32(mem, src + 0u), m01 = rd_f32(mem, src + 4u), m02 = rd_f32(mem, src + 8u),
                m03 = rd_f32(mem, src + 12u);
    const float m10 = rd_f32(mem, src + 16u), m11 = rd_f32(mem, src + 20u), m12 = rd_f32(mem, src + 24u),
                m13 = rd_f32(mem, src + 28u);
    const float m20 = rd_f32(mem, src + 32u), m21 = rd_f32(mem, src + 36u), m22 = rd_f32(mem, src + 40u),
                m23 = rd_f32(mem, src + 44u);

    const float c00 = m11 * m22 - m21 * m12;
    const float c10 = m21 * m02 - m01 * m22;
    const float c20 = m01 * m12 - m11 * m02;

    const float det = m00 * c00 + m10 * c10 + m20 * c20;
    if (det == 0.0f) {
        cpu->r[3] = 0u;
        return;
    }

    const float inv_det = 1.0f / det;
    const float c01 = m12 * m20 - m22 * m10;
    const float c11 = m22 * m00 - m02 * m20;
    const float c21 = m02 * m10 - m12 * m00;
    const float c02 = m10 * m21 - m20 * m11;
    const float c12 = m20 * m01 - m00 * m21;
    const float c22 = m00 * m11 - m10 * m01;

    const float r00 = c00 * inv_det, r01 = c10 * inv_det, r02 = c20 * inv_det;
    const float r10 = c01 * inv_det, r11 = c11 * inv_det, r12 = c21 * inv_det;
    const float r20 = c02 * inv_det, r21 = c12 * inv_det, r22 = c22 * inv_det;

    const float r03 = -(r00 * m03 + r01 * m13 + r02 * m23);
    const float r13 = -(r10 * m03 + r11 * m13 + r12 * m23);
    const float r23 = -(r20 * m03 + r21 * m13 + r22 * m23);

    wr_f32(mem, inv + 0u, r00);
    wr_f32(mem, inv + 4u, r01);
    wr_f32(mem, inv + 8u, r02);
    wr_f32(mem, inv + 12u, r03);
    wr_f32(mem, inv + 16u, r10);
    wr_f32(mem, inv + 20u, r11);
    wr_f32(mem, inv + 24u, r12);
    wr_f32(mem, inv + 28u, r13);
    wr_f32(mem, inv + 32u, r20);
    wr_f32(mem, inv + 36u, r21);
    wr_f32(mem, inv + 40u, r22);
    wr_f32(mem, inv + 44u, r23);
    cpu->r[3] = 1u;
}

void fast_guest_hooks_register() {
#if RBREW_GAME_REVISION == 272
    const char* env = std::getenv("RBREW_FAST_GUEST_HOOKS");
    const bool enabled = !(env && env[0] == '0');
    g_rbrew_fast_guest_hooks = enabled;
    if (!enabled) return;
    rbrew_set_guest_hook(0x028F5B9Cu, hook_worker_join);
    rbrew_set_guest_hook(0x02D1D730u, hook_sphere_transform);
    rbrew_set_guest_hook(0x02D1D830u, hook_sphere_merge);
    rbrew_set_guest_hook(0x02D34540u, hook_mtx44_concat);
    rbrew_set_guest_hook(0x02D349ECu, hook_mtx_concat);
    rbrew_set_guest_hook(0x02D34AD0u, hook_mtx_inverse);
#else
    g_rbrew_fast_guest_hooks = false;
#endif
}
