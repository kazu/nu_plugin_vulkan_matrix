use cubecl::wgpu::WgpuDeviceKind;
use std::sync::OnceLock;
use vulkan_matrix_validation::{Gpu, open_device};

pub fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        let kind = match std::env::var("VM_GPU").as_deref().unwrap_or("d0") {
            "d0" => WgpuDeviceKind::DiscreteGpu(0),
            "d1" => WgpuDeviceKind::DiscreteGpu(1),
            "i0" => WgpuDeviceKind::IntegratedGpu(0),
            value => panic!("unknown VM_GPU selector: {value}"),
        };
        let gpu = open_device(kind).unwrap();
        println!(
            "GPU={} compiler={} f64={}",
            gpu.name,
            gpu.client.name(),
            gpu.f64
        );
        gpu
    })
}
