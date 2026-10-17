mod support;

use cubek_matmul::{multi_level, strategy::Strategy};
use ndarray::{Array2, array};
use std::sync::Barrier;
use vulkan_matrix_validation::{download, multiply, upload};

#[test]
fn concurrent_legacy_launches_preserve_values() {
    let gpu = support::gpu();
    let start = Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            start.wait();
            for iteration in 0..64 {
                let a = array![[-2.0_f32, -1.0, 0.0], [0.0, 1.0, 2.0]];
                let b = Array2::<f32>::zeros((3, 0));
                let (empty, _) = multiply(
                    gpu,
                    &upload(gpu, a.view()).unwrap(),
                    &upload(gpu, b.view()).unwrap(),
                    &Strategy::MultiLevel(multi_level::Strategy::SimpleUnit(Default::default())),
                )
                .unwrap();
                assert_eq!(download::<f32>(gpu, &empty).unwrap().shape(), &[2, 0]);
                for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let lhs = upload(gpu, array![[value]].view()).unwrap();
                    let rhs = upload(gpu, array![[1.0_f64]].view()).unwrap();
                    let (out, _) = multiply(
                        gpu,
                        &lhs,
                        &rhs,
                        &Strategy::MultiLevel(
                            multi_level::Strategy::SimpleUnit(Default::default()),
                        ),
                    )
                    .unwrap();
                    let got = download::<f64>(gpu, &out).unwrap()[[0, 0]];
                    assert!(
                        if value.is_nan() {
                            got.is_nan()
                        } else {
                            got == value
                        },
                        "iteration={iteration} expected={value:?} actual={got:?}"
                    );
                    let lhs = upload(gpu, array![[value as f32]].view()).unwrap();
                    let rhs = upload(gpu, array![[1.0_f32]].view()).unwrap();
                    let (out, _) = multiply(
                        gpu,
                        &lhs,
                        &rhs,
                        &Strategy::MultiLevel(
                            multi_level::Strategy::SimpleUnit(Default::default()),
                        ),
                    )
                    .unwrap();
                    let got = download::<f32>(gpu, &out).unwrap()[[0, 0]];
                    assert!(
                        if value.is_nan() {
                            got.is_nan()
                        } else {
                            got as f64 == value
                        },
                        "iteration={iteration} expected={value:?} actual={got:?} f32"
                    );
                }
            }
        });
        scope.spawn(|| {
            start.wait();
            let a = Array2::from_shape_fn((31, 33), |(i, j)| ((i * 7 + j) % 5) as f32 - 2.0);
            let b = Array2::from_shape_fn((33, 65), |(i, j)| ((i + j * 3) % 5) as f32 - 2.0);
            let expected = a.dot(&b);
            for _ in 0..64 {
                let lhs = upload(gpu, a.view()).unwrap();
                let rhs = upload(gpu, b.view()).unwrap();
                let (out, _) = multiply(
                    gpu,
                    &lhs,
                    &rhs,
                    &Strategy::MultiLevel(multi_level::Strategy::SimpleUnit(Default::default())),
                )
                .unwrap();
                assert_eq!(download::<f32>(gpu, &out).unwrap(), expected);
            }
        });
    });
}
