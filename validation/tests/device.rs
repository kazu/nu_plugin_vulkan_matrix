mod support;

use cubecl::wgpu::WgpuDeviceKind;
use vulkan_matrix_validation::open_device;

#[test]
fn vulkan_setup_is_explicit_and_fallible() {
    assert!(open_device(WgpuDeviceKind::Cpu).is_err());
    assert!(open_device(WgpuDeviceKind::DiscreteGpu(usize::MAX)).is_err());
    let gpu = support::gpu();
    assert_eq!(gpu.client.name(), "wgpu<spirv>");
    assert!(gpu.f64);
    assert!(!gpu.name.to_lowercase().contains("llvmpipe"));
    println!("device={} f64={}", gpu.name, gpu.f64);
}
