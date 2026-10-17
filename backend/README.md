# nu_plugin_vulkan_matrix_backend

A project-specific Vulkan backend for [nu_plugin_vulkan_matrix](https://github.com/kazu/nu_plugin_vulkan_matrix).
CubeCL and CubeK use their published packages; this backend isolates the plugin's integration changes without global dependency patches.

## Origin and changes

Based on the published `cubecl-wgpu` 0.11.0 package from
[CubeCL v0.11.0](https://github.com/tracel-ai/cubecl/tree/v0.11.0/crates/cubecl-wgpu),
revision `52c5086d3c5e73e56b3ee4cb711bdda495a360f0`.
Original package SHA-256: `2b8c6f4eb4af39ed3bbb3e4126373d8132cbe018e669172d8d4f1c775b828497`.

- Submit queued uploads even when no compute task was recorded.
- Enable `VK_EXT_external_memory_host` and register externally owned NPY buffers.
- Add `DenormPreserve` to the public SPIR-V compiler's output for the f32/f64 types it uses.
- Use the public `DeviceHandle` API for typed server operations.

The WGSL compiler is reused from the published `cubecl-wgpu` package so its
compiler operations are registered only once, even when both backends are linked.

The plugin validates device capabilities and owns imported mappings through their GPU lifetime.
The backend retains upstream code and attribution. It is distributed under MIT; upstream's original Apache license is also retained.

## Configuration

`CUBECL_WGPU_MAX_TASKS` sets the positive number of compute tasks submitted per batch.
