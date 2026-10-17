mod support;

use cubecl::{bytes::Bytes, cube, future::block_on, prelude::*};
use support::gpu;
use vulkan_matrix_backend::RuntimeOptions;

#[cube(launch)]
fn reject_expansion(output: &mut [u32], #[comptime] refuse: bool) {
    comptime!(assert!(!refuse, "task002 rejected expansion"));
    output[0] = 1;
}

#[test]
fn invalid_runtime_configuration_is_fallible() {
    assert!(RuntimeOptions::with_tasks_max(Some(0)).is_err());
}

#[test]
fn oversized_allocation_is_rejected_before_allocating_gpu_memory() {
    let client = &gpu().client;
    let memory = &client.properties().memory;
    let requested = memory.max_page_size.checked_add(memory.alignment).unwrap();
    println!(
        "allocation limit={} rejected_size={requested}",
        memory.max_page_size
    );
    let output = client.empty(usize::try_from(requested).unwrap());
    let sync_error = block_on(client.sync_buffers([&output])).unwrap_err();
    let read_error = client.read_one(output).unwrap_err();
    for error in [sync_error, read_error] {
        assert!(
            format!("{error:?}").contains("can't allocate buffer of size:"),
            "{error:?}"
        );
    }
}

#[test]
fn launch_failure_is_reported_on_the_output_without_poisoning_other_buffers() {
    let client = &gpu().client;
    let output = client.create(Bytes::from_elems(vec![7_u32]));
    reject_expansion::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
        true,
    );
    block_on(client.sync()).unwrap();
    let error = block_on(client.sync_buffers([&output])).unwrap_err();
    assert!(format!("{error:?}").contains("task002 rejected expansion"));
    let error = client.read_one(output).unwrap_err();
    assert!(format!("{error:?}").contains("task002 rejected expansion"));
    let healthy = client.create(Bytes::from_elems(vec![11_u32]));
    assert_eq!(u32::from_bytes(&client.read_one(healthy).unwrap()), &[11]);
}
