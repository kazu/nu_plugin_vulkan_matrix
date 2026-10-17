#[path = "../tests/support/mod.rs"]
mod support;

use criterion::{Criterion, criterion_group, criterion_main};
use cubecl::{CubeElement, future::block_on, std::tensor::TensorHandle};
use ndarray::{ArrayD, Axis, IxDyn, NdFloat};
use nu_plugin_vulkan_matrix::{gpu, reduction};
use std::{hint::black_box, time::Duration};

fn measure<T: CubeElement + NdFloat + cubecl::num_traits::FromPrimitive>(
    c: &mut Criterion,
    dtype: &str,
) {
    let gpu = support::gpu();
    for n in [64, 256, 1024] {
        let input = ArrayD::from_elem(IxDyn(&[n, n]), T::one());
        let tensor = gpu::upload_view(gpu, input.view()).unwrap();
        let mut group = c.benchmark_group(format!("task006/reductions/{dtype}/{n}"));
        for (name, operation, axis) in [
            ("sum", 0, None),
            ("sum_axis0", 0, Some(0)),
            ("sum_axis1", 0, Some(1)),
            ("max", 1, None),
            ("max_axis0", 1, Some(0)),
            ("max_axis1", 1, Some(1)),
            ("mean", 2, None),
        ] {
            let run = |tensor: &TensorHandle| match operation {
                0 => reduction::sum(gpu, tensor, axis).unwrap(),
                1 => reduction::max(gpu, tensor, axis).unwrap(),
                _ => reduction::mean(gpu, tensor).unwrap(),
            };
            let expected = if operation == 0 {
                T::from(if axis.is_some() { n } else { n * n }).unwrap()
            } else {
                T::one()
            };
            assert!(
                gpu::download_array::<T>(gpu, &run(&tensor))
                    .unwrap()
                    .iter()
                    .all(|&value| value == expected)
            );
            group.bench_function(format!("{name}/cpu"), |bench| {
                bench.iter(|| match operation {
                    0 => {
                        if let Some(axis) = axis {
                            black_box(input.sum_axis(Axis(axis)));
                        } else {
                            black_box(input.sum());
                        }
                    }
                    1 => {
                        if let Some(axis) = axis {
                            black_box(
                                input.fold_axis(Axis(axis), T::neg_infinity(), |&a, &b| a.max(b)),
                            );
                        } else {
                            black_box(input.fold(T::neg_infinity(), |a, &b| a.max(b)));
                        }
                    }
                    _ => {
                        black_box(input.mean().unwrap());
                    }
                })
            });
            group.bench_function(format!("{name}/resident_wait"), |bench| {
                bench.iter(|| {
                    let output = run(&tensor);
                    block_on(gpu.client.sync_buffers([&output.handle])).unwrap();
                    black_box(output);
                })
            });
            group.bench_function(format!("{name}/resident_readback"), |bench| {
                bench.iter(|| black_box(gpu::download_array::<T>(gpu, &run(&tensor)).unwrap()))
            });
            group.bench_function(format!("{name}/end_to_end"), |bench| {
                bench.iter(|| {
                    let uploaded = gpu::upload_view(gpu, input.view()).unwrap();
                    black_box(gpu::download_array::<T>(gpu, &run(&uploaded)).unwrap());
                })
            });
        }
        group.finish();
    }
}

fn reductions(c: &mut Criterion) {
    measure::<f64>(c, "f64");
    measure::<f32>(c, "f32");
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500));
    targets = reductions
}
criterion_main!(benches);
