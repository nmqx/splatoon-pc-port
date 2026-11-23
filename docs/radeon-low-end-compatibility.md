# AMD Radeon & Low-End PC Compatibility

This document summarizes the rendering, shader, Vulkan memory, CPU ISA, and
Release logging issues diagnosed on AMD Radeon integrated GPUs (verified on
Radeon 890M, RDNA 3.5, Windows Vulkan driver / LLPC `24.30.22.03`) and older or
low-VRAM PCs, together with the implemented fixes and configuration knobs.

## Diagnosed root causes

### AMD Vulkan & GX2 rendering hazards

- **Depth-copy failure on `D32_SFLOAT_S8_UINT` (`>720p`):** Radeon 890M does not
  advertise `VK_FORMAT_D24_UNORM_S8_UINT`, so depth-stencil buffers fall back to
  `VK_FORMAT_D32_SFLOAT_S8_UINT`, which lacks `VK_FORMAT_FEATURE_BLIT_DST_BIT`.
  Scaled depth-array alias copies and scaled `GX2CopySurface` depth blits
  silently returned without copying (while `GX2CopySurface` still called
  `note_gpu_write()` and dropped `VK_IMAGE_ASPECT_STENCIL_BIT` on unscaled
  depth-stencil copies).
- **`QUADS` / `QUAD_STRIP` geometry-shader path:** Emitting `QUADS` (`0x13`) and
  `QUAD_STRIP` (`0x14`) through `LINE_LIST_WITH_ADJACENCY` and an assembly
  geometry shader introduced driver/LLPC fragility on AMD PAL compared to direct
  triangle expansion.
- **Push-descriptor state tracking & missing fallback:** Partial
  `vkCmdPushDescriptorSetKHR` subset writes across pipeline or render-pass
  transitions could leave stale descriptors bound, and `depth_to_color_pass`
  used push descriptors unconditionally without checking
  `gx2vk::caps().push_descriptor`.
- **Intra-pass attachment feedback loops:** Draws that sampled a texture
  aliasing an active color or depth attachment did not end the active render
  pass and insert a graphics barrier beforehand.
- **Mutable image format lists (RDNA DCC):** Images created with
  `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` did not chain
  `VkImageFormatListCreateInfo`, causing AMD RDNA drivers to disable or
  mishandle DCC metadata across sRGB/UNORM and compatible view aliases.
- **Barrier & overlay synchronization ordering:** `barrier_for()` snapshotted
  pending write/WAR masks before calling `leave_rendering()`; `present()` could
  skip `leave_rendering()` when `RBREW_SCAN_COPY=0`; and the ImGui overlay's
  `TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL` barrier omitted
  `VK_ACCESS_COLOR_ATTACHMENT_READ_BIT` required by `LOAD_OP_LOAD` and blending.
- **Custom border color without format:** `s_caps.custom_border_color` checked
  `customBorderColors` without verifying `customBorderColorWithoutFormat`, even
  though `sampler()` sets `border.format = VK_FORMAT_UNDEFINED`.
- **Shared texture swizzle clobber & uninitialized mip 0:** Binding the same
  texture or render target to multiple slots with different `word4` swizzles
  overwrote the shared `TexObject::swizzle`, and inaccessible/unuploaded mip-0
  textures remained uninitialized VRAM.
- **Per-target MRT blend state & untracked VBO/IBO staleness:** MRT blend
  registers (`CB_BLEND0..7_CONTROL`, `CB_COLOR_CONTROL`) were collapsed to
  target-0 state, and `kGx2WriteUntracked` short-circuited mid-frame guest
  VBO/IBO updates on matching epochs.

### Weather & post-FX shaders

- **Uninitialized `s_dome` VRAM:** The `4096x1024 RGBA16F` cloud dome was never
  cleared on creation or resize. At `shader_quality=3`, up to 63/64ths of texels
  remained uninitialized VRAM for up to 64 frames while linearly sampled into
  `exposedCloud` (`+Inf` -> `NaN`), bloom, sky reflections, and TAA.
- **Unguarded NaN/Inf math:** `kAmbientOcclusion` called
  `normalize(cross(dx, dy))` without checking for a zero-length cross product,
  `fxaa` took `sqrt(luminance(...))` without clamping negative/non-finite HDR
  inputs, `moonDisc` called `cross(m, vec3(0, 1, 0))` without a zenith fallback,
  and `exposedCloud`/`cloudsAt` did not sanitize non-finite values.

### Latte GLSL decompiler & translator

- **Uninitialized shader locals (`bool` UB on LLPC):** `activeMaskStack`
  (`depth + 1`) and `activeMaskStackC` (`depth + 2`) were only initialized up to
  `< depth` and `< depth + 1`, leaving the tail element uninitialized. Vertex
  `attrDecoder` and cubemap `cubeMapSTM` / `cubeMapFaceId` were also emitted
  without initializers.
- **GLSL `#extension` directive ordering:** Vertex shader translation inserted
  `layout(push_constant) uniform RbrewPushConstants` before `#extension` lines,
  violating GLSL §3.3.

### Low-VRAM iGPU memory pressure & CPU accessibility

- **Split-heap iGPU BAR exhaustion:** Radeon 890M exposes three memory heaps
  (`256 MiB DEVICE_LOCAL`, `256 MiB DEVICE_LOCAL|HOST_VISIBLE` BAR, and
  `15.56 GiB HOST_VISIBLE|HOST_COHERENT` shared system RAM). Ranking the small
  BAR above shared coherent RAM for `MemoryUse::Upload`—combined with `64 MiB`
  allocator slabs, `32 MiB` upload chunks (up to `512 MiB` before trimming),
  `4096x1024` weather dome, VBO/IBO VRAM promotion, a large vertex cache, and
  `cores - 2` (22 on a 24-thread CPU) shader compile workers—exhausted local
  memory and caused WDDM thrashing and AMD LLPC contention.
- **Mandatory AVX2 codegen:** Unconditional `/arch:AVX2` (MSVC) and
  `-march=native` (GCC/Clang) produced binaries that crash with illegal
  instruction faults on older x86-64 CPUs lacking AVX2.
- **Silent Release failures:** `RBREW_DISABLE_LOGGING` compiled out
  `RBREW_LOG_ERROR` and `RBREW_LOG_FATAL` (including `RBREW_SHADER_ERRORS=1/2`),
  hiding Vulkan initialization, shader compilation, and OOM errors in Release
  builds.

---

## Implemented fixes by subsystem

### 1. Vulkan core & low-VRAM iGPU allocator

Files: `recompiler/port/os/gx2/vk/gx2_vk.h`, `recompiler/port/os/gx2/vk/gx2_vk.cpp`,
`recompiler/port/os/gx2/vk/gx2_vk_memory.h`, `recompiler/port/os/gx2/vk/gx2_imgui_overlay.cpp`

- **Device capability detection:** Added `is_amd`, `is_igpu`, `low_vram_igpu`,
  and `device_local_bytes` to `gx2vk::Caps` (reset cleanly on `shutdown()`).
  `low_vram_igpu` activates when `DEVICE_LOCAL` capacity is `<= 1 GiB` on an
  integrated GPU or when `<= 1 GiB` `DEVICE_LOCAL` is paired with a larger
  shared `HOST_VISIBLE | HOST_COHERENT` heap.
- **Shared-heap upload ranking & smaller pools:** Added
  `memory_detail::prefer_shared_upload()` and an optional `prefer_shared_upload`
  parameter to `memory_detail::memory_types()` so split-heap low-VRAM iGPUs
  place `MemoryUse::Upload` rings in shared `HOST_VISIBLE | HOST_COHERENT`
  memory (uncached WC preferred, then cached) rather than saturating a small
  BAR, while keeping `DeviceLocal` preferring `DEVICE_LOCAL` with shared
  fallback and preserving existing behavior on discrete GPUs and single-heap
  UMA. When `low_vram_igpu` is active, slab `pool_block_size()` scales from
  `64 MiB` to `16 MiB` and post-burst upload trim ceilings tighten from `16/12`
  to `12/8`, while `kChunkSize` (`32 MiB`) and `kChunkReclaimThreshold` (`10`)
  remain unchanged because upload chunks route to shared system RAM via
  `s_prefer_shared_upload`, preserving the 8-chunk residency contract.
- **Mutable image format lists:** `create_image()` chains core Vulkan 1.2/1.3
  `VkImageFormatListCreateInfo` for `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` images
  (`mutable_view_formats()` pairs compatible formats such as
  `R8G8B8A8_SRGB`/`R8G8B8A8_UNORM` and BC/integer aliases, filtered by
  `format_supports()` for the requested usage) to preserve AMD RDNA DCC.
- **Correctness & synchronization fixes:**
  - Gated `s_caps.custom_border_color` on both `customBorderColors` and
    `customBorderColorWithoutFormat`.
  - Updated `barrier_for()` to invoke `leave_rendering()` before snapshotting
    `s_pending_write` / `s_pending_war`, and ensured `present()` calls
    `leave_rendering()` unconditionally.
  - Added `VK_ACCESS_COLOR_ATTACHMENT_READ_BIT` to the ImGui overlay's
    `TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL` barrier and explicit
    `COLOR_ATTACHMENT_WRITE_BIT -> TRANSFER_WRITE_BIT` on the return barrier.

### 2. GX2 render surfaces, draw pipeline, `QUADS`, descriptors & depth copy

Files: `recompiler/port/os/gx2/vk/gx2_render_vk_state.h`,
`recompiler/port/os/gx2/gx2_render_vk.cpp`, `recompiler/port/os/gx2/gx2_render_vk_draw.cpp`

- **CPU `TRIANGLE_LIST` expansion for `QUADS` / `QUAD_STRIP`:** Added
  `append_quad_triangles()` emitting `{a, b, d, b, c, d}` for `QUADS` (`0x13`)
  and `{a, b, d, c, a, d}` for `QUAD_STRIP` (`0x14`) (preserving CCW winding and
  last-vertex provoking vertex `d`) in both `gx2_render_draw()` and
  `gx2_render_draw_indexed()`, mapping both topologies to
  `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST` and leaving `RECTS` (`0x11`) untouched.
- **Full push-descriptor writes & fallback:** Invalidated `s_bindings`
  descriptor state on `end_rendering()` and pipeline switches, switched
  `bind_descriptors()` to push the full `write_count` when dirty/invalidated,
  added the `RBREW_PUSH_DESCRIPTORS=0` runtime gate (`push_descriptors_enabled()`),
  and gave both `depth_to_color_pass` and `depth_to_depth_pass` a shared
  descriptor-pool fallback when push descriptors are disabled or unsupported.
- **Depth-to-depth shader pass & stencil copy:** Implemented
  `gx2r::depth_to_depth_pass()` (nearest `texelFetch` writing `gl_FragDepth`,
  with a scratch depth-attachment fallback if the destination lacks
  `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`). Scaled depth-array alias
  copies and scaled `GX2CopySurface` depth copies now fall back to
  `depth_to_depth_pass()` whenever `BLIT_SRC | BLIT_DST` is unsupported, only
  mark the destination written when the copy succeeds, and include
  `VK_IMAGE_ASPECT_STENCIL_BIT` on unscaled depth-stencil `vkCmdCopyImage`.
- **Feedback-loop barrier:** `build_stage()` detects when any bound texture
  aliases an active render-pass color or depth attachment
  (`StageDescriptors::aliases_active_fb`), triggering `end_rendering()` and
  `gx2vk::barrier_for(gx2vk::kSyncGraphics)` before resuming rendering.
- **Per-slot swizzle, mip-0 zeroing, MRT blend & VBO/IBO coherency:**
  - Stored per-slot `word4` on `BoundTexture::swizzle` and added a 5-argument
    `gx2r::tex_object_view(..., swizzle_override)` overload so multiple units
    never clobber shared `TexObject::swizzle`.
  - Zero-cleared any non-depth texture whose mip 0 could not be uploaded.
  - Decoded `CB_BLEND0..7_CONTROL` (`0xA1E0..0xA1E7`) and `CB_COLOR_CONTROL`
    into `RasterState::target_blends[0..7]` and wired them through pipeline
    hashing and creation.
  - Removed the `kGx2WriteUntracked` epoch short-circuit for VBOs and IBOs so
    mid-frame guest updates are always hash-checked.
- **Low-end / AMD iGPU tuning:** Capped shader compile and warm-start workers to
  `<= 4` on `is_amd || low_vram_igpu`; defaulted `RBREW_VBO_VRAM` and
  `RBREW_IBO_VRAM` to OFF on `low_vram_igpu` (still overridable by environment
  variable); and capped `vertex_cache_limits()` to `<= 64 MiB` and `<= 32768`
  entries on `low_vram_igpu`.

### 3. Weather & post-FX shaders

Files: `recompiler/port/os/gx2/gx2_weather.cpp`, `recompiler/port/os/gx2/gx2_weather_shaders.h`

- Added `VK_IMAGE_USAGE_TRANSFER_DST_BIT` to `s_dome` and cleared it to
  `{0.0, 0.0, 0.0, 1.0}` (zero radiance, full transmittance) on creation and
  size/quality transitions with proper transfer/compute synchronization.
- Capped `s_dome` resolution to `2048x512` (instead of `4096x1024`) at
  `quality=3` on `low_vram_igpu`, saving ~24 MiB of `DEVICE_LOCAL` memory.
- Added NaN/Inf and degenerate-vector guards in `kAmbientOcclusion`,
  `exposedCloud` / `cloudsAt`, `fxaa` (`srcLuma`), and `moonDisc`.

### 4. Latte GLSL decompiler & translator

Files: `recompiler/port/third_party/cemu_shader/Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompilerEmitGLSL.cpp`,
`recompiler/port/os/gx2/r700_to_glsl.cpp`

- Fixed `activeMaskStack` (`depth + 1`) and `activeMaskStackC` (`depth + 2`)
  initialization loops so every declared element is initialized, and
  zero-initialized `attrDecoder = uvec4(0u)`, `cubeMapSTM = vec3(0.0)`, and
  `cubeMapFaceId = 0`.
- Updated vertex-shader translation in `r700_to_glsl.cpp` to insert
  `layout(push_constant) uniform RbrewPushConstants` strictly after `#version`
  and all `#extension` directives (GLSL §3.3).

### 5. Build portability & Release error logging

Files: `recompiler/port/CMakeLists.txt`, `recompiler/include/rebrewu/logger.hpp`

- Added CMake option `RBREW_ENABLE_AVX2` (default `OFF`) gating `/arch:AVX2`
  across MSVC targets (`gambit_generated`, `gambit_runtime`, `cemu_shader`,
  `gambit_guest_bench`) and `-mavx2 -mfma` / `-mavx2 -mfma -mbmi2 -mmovbe`
  (replacing unconditional `-march=native`) on GCC/Clang so default builds run
  on baseline x86-64 CPUs without AVX2.
- Updated `logger.hpp` under `RBREW_DISABLE_LOGGING` so verbose macros
  (`RBREW_LOG_DEBUG`, `RBREW_LOG_INFO`, `RBREW_LOG_WARN`, their `_BEGIN`
  variants, `RBREW_LOG_RAW`, and `RBREW_LOG_STREAM`) remain compiled out with
  zero hot-path cost, while `RBREW_LOG_ERROR`, `RBREW_LOG_FATAL`,
  `RBREW_LOG_ERROR_BEGIN`, and `RBREW_LOG_FATAL_BEGIN` remain active in Release
  builds (restoring Vulkan, OOM, and `RBREW_SHADER_ERRORS=1/2` error output).

---

## Build & runtime knobs

| Knob | Type | Default | Description |
| --- | --- | --- | --- |
| **`RBREW_ENABLE_AVX2`** | CMake option | `OFF` | Opt-in AVX2/FMA/BMI2/MOVBE codegen on x64 builds. Leave `OFF` for portable binaries that run on older non-AVX2 CPUs; set `-DRBREW_ENABLE_AVX2=ON` for Haswell+ / Zen+ local builds. |
| **`RBREW_DIAGNOSTIC_RUNTIME`** | CMake option | `OFF` | Keeps verbose `DEBUG`/`INFO`/`WARN` logs, frame capture, and profiler hooks. When `OFF`, `RBREW_DISABLE_LOGGING` strips verbose diagnostics while still preserving `ERROR` and `FATAL` logs. |
| **`RBREW_PUSH_DESCRIPTORS`** | Env var | `1` (if supported) | Set to `0` to disable `VK_KHR_push_descriptor` and force descriptor-pool allocation across both main draws and depth/color transfer passes. |
| **`RBREW_SHADER_ERRORS`** | Env var | unset | Works in Release builds now that `RBREW_LOG_ERROR` is preserved: `1` logs glslang compiler diagnostics; `2` also dumps the failing GLSL source to `stderr`. |
| **`RBREW_VBO_VRAM`** / **`RBREW_IBO_VRAM`** | Env var | `1` (discrete/large UMA), `0` (`low_vram_igpu`) | Controls staging vertex/index buffers into `DEVICE_LOCAL` memory. Defaults to `0` on low-VRAM iGPUs to avoid local-heap exhaustion; set explicitly to `1` or `0` to override. |
| **`RBREW_LOG_LEVEL`** | Env var | `debug` | Runtime minimum log level (`debug`, `info`, `warning`, `error`, `fatal`, `off`). In Release builds (`RBREW_DISABLE_LOGGING`), setting `off` silences even `ERROR`/`FATAL` output. |

---

## Regression test coverage

Configure `recompiler/port` with `-DRBREW_BUILD_TESTS=ON` and run CTest:

- **`memory_policy`, `memory_allocator`, `memory_transfers`**
  (`recompiler/tests/test_vulkan_memory.cpp`): Models Radeon 890M's 3-heap
  layout (`256 MiB DEVICE_LOCAL`, `256 MiB BAR`, `15.56 GiB shared`) and
  verifies `MemoryUse::Upload` selects the shared coherent heap while
  `DeviceLocal` prefers `DEVICE_LOCAL` with shared fallback.
- **`weather_gpu`** (`recompiler/tests/test_weather.cpp`): Exercises
  `exposedCloud` NaN/`±Inf`/negative inputs, `kAmbientOcclusion` collinear-depth
  zero-cross inputs, `kSkyVertex`/`kSkyFragment` with a zenith moon and
  non-finite cloud dome texels, `fxaa` negative-RGB inputs, and `s_dome`
  frame-0 `quality=3` and quality-transition clears.
- **`shader_translation`** (`recompiler/tests/test_shader_performance.cpp`):
  Verifies every translated vertex shader contains `layout(push_constant)` and
  places all `#extension` directives strictly before it.
- **`render_geometry`** (and standalone `gambit_render_bench quads_depth`,
  `recompiler/tests/test_render_performance.cpp`): Verifies `QUADS`/`QUAD_STRIP`
  triangle expansion and live GPU `QUADS` rasterization, non-mutating per-slot
  `swizzle_override`, and `depth_to_depth_pass` across `D32_SFLOAT` (direct and
  non-attachable scratch fallback) and `D32_SFLOAT_S8_UINT`.
