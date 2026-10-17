use cubecl::wgpu::{
    AutoCompiler, RuntimeOptions, Vulkan, WgpuBackend, WgpuDevice, WgpuDeviceKind, WgpuRuntime,
    try_init_setup,
};
use cubecl::{bytes::Bytes, future::block_on};
use cubecl_runtime::runtime::Runtime;

#[test]
fn logical_device_destruction_is_reported_by_sync_and_readback() {
    let device = WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0)).on(WgpuBackend::Vulkan);
    let setup = try_init_setup::<Vulkan>(&device, RuntimeOptions::try_default().unwrap()).unwrap();
    let client = WgpuRuntime::<AutoCompiler>::client(&device);
    let output = client.create(Bytes::from_elems(vec![1_u32]));
    block_on(client.sync_buffers([&output])).unwrap();
    setup.device.destroy();
    let poll = setup.device.poll(cubecl::wgpu::wgpu::PollType::Poll);
    let sync = block_on(client.sync());
    let read = client.read_one(output);
    println!("logical device destroyed: poll={poll:?} sync={sync:?} read={read:?}");
    assert!(sync.is_err());
    assert!(read.is_err());
}
