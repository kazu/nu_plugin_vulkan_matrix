#[path = "../tests/support/mod.rs"]
mod support;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use cubecl::{CubeElement, future::block_on};
use ndarray::{ArrayD, IxDyn, NdFloat, s};
use nu_plugin_vulkan_matrix::gpu::{download_array, upload_array, upload_view};
use std::{hint::black_box, time::Duration};

fn measure<T: CubeElement + NdFloat>(c: &mut Criterion, dtype: &str) {
    let gpu = support::gpu();
    let standard = ArrayD::from_shape_fn(IxDyn(&[64, 128, 128]), |i| {
        T::from(((i[0] * 17 + i[1] * 7 + i[2]) % 32) as f64 / 32.0).unwrap()
    });
    let negative = standard.clone().slice_move(s![.., .., ..;-1]).into_dyn();
    let mut group = c.benchmark_group(format!("task003/transfers/{dtype}"));
    for (layout, array) in [("standard", &standard), ("negative", &negative)] {
        assert_eq!(array.clone().strides(), array.strides());
        for tensor in [
            upload_view(gpu, array.view()).unwrap(),
            upload_array(gpu, array.clone()).unwrap(),
        ] {
            assert_eq!(download_array::<T>(gpu, &tensor).unwrap(), *array);
        }
        group.bench_function(format!("borrowed/{layout}"), |bench| {
            bench.iter(|| {
                let tensor = upload_view(gpu, array.view()).unwrap();
                block_on(gpu.client.sync_buffers([&tensor.handle])).unwrap();
                black_box(tensor);
            })
        });
        group.bench_function(format!("owned/{layout}"), |bench| {
            bench.iter_batched(
                || array.clone(),
                |array| {
                    let tensor = upload_array(gpu, array).unwrap();
                    block_on(gpu.client.sync_buffers([&tensor.handle])).unwrap();
                    black_box(tensor);
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn transfers(c: &mut Criterion) {
    measure::<f64>(c, "f64");
    measure::<f32>(c, "f32");
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10)
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_millis(500)).without_plots();
    targets = transfers
}
criterion_main!(benches);
