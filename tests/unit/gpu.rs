use crate::gpu::{Gpu, open_device};
use cubecl::wgpu::WgpuDeviceKind;
use std::sync::{Arc, OnceLock};

pub fn gpu() -> &'static Arc<Gpu> {
    static GPU: OnceLock<Arc<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| Arc::new(open_device(WgpuDeviceKind::DiscreteGpu(0)).unwrap()))
}
