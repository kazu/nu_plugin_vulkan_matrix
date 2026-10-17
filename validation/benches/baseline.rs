#[path = "../tests/support/mod.rs"]
mod support;

use criterion::{Criterion, criterion_group, criterion_main};
use cubecl::{CubeElement, future::block_on};
use cubek_matmul::{multi_level, strategy::Strategy};
use ndarray::{Array2, NdFloat};
use nu_plugin_vulkan_matrix::matmul::multiply as resident_multiply;
use std::hint::black_box;
use std::time::{Duration, Instant};
use vulkan_matrix_validation::{download, multiply, upload};

fn measure<T: CubeElement + NdFloat>(c: &mut Criterion, dtype: &str) {
    let gpu = support::gpu();
    let mut group = c.benchmark_group(format!("task002/{dtype}"));
    for n in [64, 256, 1024] {
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            T::from(((i * n + j) % 17) as f64 / 16.0).unwrap()
        });
        let b = Array2::from_shape_fn((n, n), |(i, j)| {
            T::from(((i * 7 + j * 11) % 17) as f64 / 16.0).unwrap()
        });
        let expected = a.dot(&b);
        let lhs = upload(gpu, a.view()).unwrap();
        let rhs = upload(gpu, b.view()).unwrap();
        block_on(gpu.client.sync_buffers([&lhs.handle, &rhs.handle])).unwrap();
        for (name, strategy) in [
            ("naive", Strategy::MultiLevel(multi_level::Strategy::Naive)),
            (
                "simple",
                Strategy::MultiLevel(multi_level::Strategy::SimpleUnit(Default::default())),
            ),
            ("auto", Strategy::Auto),
        ] {
            let start = Instant::now();
            let (out, elems) = multiply(gpu, &lhs, &rhs, &strategy).unwrap();
            block_on(gpu.client.sync_buffers([&out.handle])).unwrap();
            eprintln!(
                "first_for_case dtype={dtype} n={n} strategy={name} elapsed_ns={} elems={elems:?}",
                start.elapsed().as_nanos()
            );
            assert_eq!(download::<T>(gpu, &out).unwrap(), expected);
            group.bench_function(format!("{name}/resident_wait/{n}"), |bench| {
                bench.iter(|| {
                    let (out, _) = multiply(gpu, &lhs, &rhs, &strategy).unwrap();
                    block_on(gpu.client.sync_buffers([&out.handle])).unwrap();
                })
            });
            group.bench_function(format!("{name}/resident_readback/{n}"), |bench| {
                bench.iter(|| {
                    let (out, _) = multiply(gpu, &lhs, &rhs, &strategy).unwrap();
                    black_box(download::<T>(gpu, &out).unwrap())
                })
            });
            group.bench_function(format!("{name}/end_to_end/{n}"), |bench| {
                bench.iter(|| {
                    let lhs = upload(gpu, a.view()).unwrap();
                    let rhs = upload(gpu, b.view()).unwrap();
                    let (out, _) = multiply(gpu, &lhs, &rhs, &strategy).unwrap();
                    black_box(download::<T>(gpu, &out).unwrap())
                })
            });
            group.bench_function(format!("{name}/metadata/{n}"), |bench| {
                bench.iter(|| black_box((out.shape(), out.strides(), out.dtype)))
            });
        }
        let identity = Array2::<T>::eye(n);
        let third = upload(gpu, identity.view()).unwrap();
        let product = resident_multiply(gpu, &lhs, &rhs).unwrap();
        let chain = resident_multiply(gpu, &product, &third).unwrap();
        assert_eq!(download::<T>(gpu, &chain).unwrap(), expected);
        group.bench_function(format!("product/resident_wait/{n}"), |bench| {
            bench.iter(|| {
                let out = resident_multiply(gpu, &lhs, &rhs).unwrap();
                block_on(gpu.client.sync_buffers([&out.handle])).unwrap();
            })
        });
        group.bench_function(format!("product/end_to_end/{n}"), |bench| {
            bench.iter(|| {
                let left = upload(gpu, a.view()).unwrap();
                let right = upload(gpu, b.view()).unwrap();
                let out = resident_multiply(gpu, &left, &right).unwrap();
                black_box(download::<T>(gpu, &out).unwrap())
            })
        });
        group.bench_function(format!("product/chain3_wait/{n}"), |bench| {
            bench.iter(|| {
                let ab = resident_multiply(gpu, &lhs, &rhs).unwrap();
                let abc = resident_multiply(gpu, &ab, &third).unwrap();
                block_on(gpu.client.sync_buffers([&abc.handle])).unwrap();
            })
        });
        group.bench_function(format!("product/chain3_end_to_end/{n}"), |bench| {
            bench.iter(|| {
                let left = upload(gpu, a.view()).unwrap();
                let right = upload(gpu, b.view()).unwrap();
                let third = upload(gpu, identity.view()).unwrap();
                let ab = resident_multiply(gpu, &left, &right).unwrap();
                let abc = resident_multiply(gpu, &ab, &third).unwrap();
                black_box(download::<T>(gpu, &abc).unwrap())
            })
        });
        let transposed = upload(gpu, a.t()).unwrap();
        let expected_transposed = a.t().dot(&b);
        let out = resident_multiply(gpu, &transposed, &rhs).unwrap();
        assert_eq!(download::<T>(gpu, &out).unwrap(), expected_transposed);
        group.bench_function(format!("product/transpose_wait/{n}"), |bench| {
            bench.iter(|| {
                let out = resident_multiply(gpu, &transposed, &rhs).unwrap();
                block_on(gpu.client.sync_buffers([&out.handle])).unwrap();
            })
        });
    }
    group.finish();
}

fn baseline(c: &mut Criterion) {
    measure::<f64>(c, "f64");
    measure::<f32>(c, "f32");
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500)).without_plots();
    targets = baseline
}
criterion_main!(benches);
