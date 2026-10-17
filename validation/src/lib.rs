//! Vulkan kernel validation, including explicit denormal preservation.

pub use nu_plugin_vulkan_matrix::gpu::{Gpu, Result, download, open_device, upload};
pub use nu_plugin_vulkan_matrix::matmul::launch_multiply as multiply;
