#pragma once
#include <cstdint>

struct CPUState;

extern "C" {
extern bool g_rbrew_fast_guest_hooks;

void rbrew_fast_v272_worker_join(CPUState* __restrict cpu, uint8_t* __restrict mem);
void rbrew_fast_v272_sphere_transform(CPUState* __restrict cpu, uint8_t* __restrict mem);
void rbrew_fast_v272_sphere_merge(CPUState* __restrict cpu, uint8_t* __restrict mem);
void rbrew_fast_v272_mtx44_concat(CPUState* __restrict cpu, uint8_t* __restrict mem);
void rbrew_fast_v272_mtx_concat(CPUState* __restrict cpu, uint8_t* __restrict mem);
void rbrew_fast_v272_mtx_inverse(CPUState* __restrict cpu, uint8_t* __restrict mem);
}

void fast_guest_hooks_register();
