//! GPU-resident arrays with explicit host transfers.

extern crate self as vulkan_matrix_backend;
include!("../backend/src/lib.rs");

mod convert;
mod elementwise;
#[cfg(target_os = "linux")]
mod file_io;
pub mod gpu;
pub mod layout;
pub mod matmul;
pub mod plugin;
mod reduce_kernel;
pub mod reduction;
mod resident;

#[cfg(test)]
#[path = "../tests/unit/resident.rs"]
mod resident_tests;

#[cfg(test)]
#[path = "../tests/unit/convert.rs"]
mod convert_tests;

#[cfg(test)]
#[path = "../tests/unit/elementwise.rs"]
mod elementwise_tests;

#[cfg(test)]
#[path = "../tests/unit/gpu.rs"]
mod test_gpu;
