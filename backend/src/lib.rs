#[macro_use]
extern crate derive_new;

extern crate alloc;

mod backend;
mod compiler;
#[cfg(feature = "spirv")]
mod float_controls;
mod server_adapter;
pub use server_adapter::run_with_server;
mod compute;
mod device;
mod graphics;
mod runtime;

pub use compiler::base::*;
pub use compute::*;
pub use device::*;
pub use graphics::*;
pub use runtime::*;
/// The exact wgpu version used by this runtime, for sharing resources with applications.
pub use wgpu;

#[cfg(feature = "spirv")]
pub use backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub use backend::metal;
