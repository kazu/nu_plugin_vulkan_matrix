mod support;

use cubecl::{CubeElement, future::block_on};
use cubek_matmul::{definition::MatmulElems, multi_level, strategy::Strategy};
use ndarray::{Array2, ArrayView2, array, s};
use support::gpu;
use vulkan_matrix_validation::{download, multiply, upload};

fn strategies() -> [Strategy; 3] {
    [
        Strategy::MultiLevel(multi_level::Strategy::Naive),
        Strategy::MultiLevel(multi_level::Strategy::SimpleUnit(Default::default())),
        Strategy::Auto,
    ]
}

fn assert_precision<T: CubeElement>(elems: &MatmulElems) {
    for elem in [
        elems.lhs_global,
        elems.rhs_global,
        elems.acc_global,
        elems.lhs_stage,
        elems.rhs_stage,
        elems.acc_stage,
        elems.lhs_register,
        elems.rhs_register,
        elems.acc_register,
    ] {
        assert_eq!(elem, T::cube_type(), "{elems:?}");
    }
}

#[test]
fn f64_sentinel_survives_and_inputs_can_drop_before_completion() {
    let a = array![[16777217.0_f64, 1.0, -16777216.0], [2.0, 3.0, 4.0]];
    let b = array![[1.0_f64, 0.0], [1.0, 1.0], [1.0, 0.0]];
    let expected = a.dot(&b);
    assert_eq!(expected[[0, 0]], 2.0);
    for strategy in strategies() {
        let lhs = upload(gpu(), a.view()).unwrap();
        let rhs = upload(gpu(), b.view()).unwrap();
        let (out, elems) = multiply(gpu(), &lhs, &rhs, &strategy).unwrap();
        println!("strategy={strategy} elems={elems:?}");
        assert_precision::<f64>(&elems);
        drop((lhs, rhs));
        block_on(gpu().client.sync_buffers([&out.handle])).unwrap();
        assert_eq!(download::<f64>(gpu(), &out).unwrap(), expected);
    }
}

#[test]
fn f32_and_f64_agree_with_ndarray_at_boundaries() {
    for (m, k, n) in [
        (1, 1, 1),
        (2, 3, 4),
        (31, 33, 17),
        (32, 32, 32),
        (33, 31, 35),
    ] {
        let a = Array2::from_shape_fn((m, k), |(i, j)| ((i * k + j) as f64 * 0.7).sin());
        let b = Array2::from_shape_fn((k, n), |(i, j)| ((i * n + j) as f64 * 0.3).cos());
        let a32 = a.mapv(|x| x as f32);
        let b32 = b.mapv(|x| x as f32);
        for strategy in strategies() {
            let (out, elems) = multiply(
                gpu(),
                &upload(gpu(), a.view()).unwrap(),
                &upload(gpu(), b.view()).unwrap(),
                &strategy,
            )
            .unwrap();
            assert_precision::<f64>(&elems);
            let actual = download::<f64>(gpu(), &out).unwrap();
            let expected = a.dot(&b);
            let abs_sum = a.mapv(f64::abs).dot(&b.mapv(f64::abs));
            let ku = k as f64 * (f64::EPSILON / 2.0);
            for ((actual, expected), sum) in actual.iter().zip(expected.iter()).zip(abs_sum.iter())
            {
                assert!(
                    (actual - expected).abs() <= 2.0 * ku / (1.0 - ku) * sum,
                    "f64 {m}x{k}x{n} {strategy}: {actual} != {expected}"
                );
            }
            let (out, elems) = multiply(
                gpu(),
                &upload(gpu(), a32.view()).unwrap(),
                &upload(gpu(), b32.view()).unwrap(),
                &strategy,
            )
            .unwrap();
            assert_precision::<f32>(&elems);
            let actual = download::<f32>(gpu(), &out).unwrap();
            let expected = a32.dot(&b32);
            let abs_sum = a32
                .mapv(|v| f64::from(v).abs())
                .dot(&b32.mapv(|v| f64::from(v).abs()));
            let ku = k as f64 * (f64::from(f32::EPSILON) / 2.0);
            for ((actual, expected), sum) in actual.iter().zip(expected.iter()).zip(abs_sum.iter())
            {
                assert!(
                    (f64::from(*actual) - f64::from(*expected)).abs()
                        <= 2.0 * ku / (1.0 - ku) * sum,
                    "f32 {m}x{k}x{n} {strategy}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn incompatible_shapes_return_an_error() {
    let lhs = upload(gpu(), array![[1.0_f64, 2.0]].view()).unwrap();
    let rhs = upload(gpu(), array![[1.0_f64, 2.0]].view()).unwrap();
    assert!(multiply(gpu(), &lhs, &rhs, &strategies()[0]).is_err());
}

#[test]
fn strided_inputs_multiply_like_ndarray() {
    let values = Array2::from_shape_fn((6, 7), |(i, j)| (i * 7 + j) as f64 / 8.0);
    check_layouts(&values);
    check_layouts(&values.mapv(|x| x as f32));
}

fn check_layouts<T: CubeElement + ndarray::NdFloat>(values: &Array2<T>) {
    for view in [
        values.view(),
        values.t(),
        values.slice(s![1..5, ..]),
        values.slice(s![..;2, ..;3]),
        values.slice(s![..;-1, ..]),
    ] {
        let rhs = Array2::from_shape_fn((4, view.ncols()), |(i, j)| {
            T::from((i + j) as f64 / 8.0).unwrap()
        });
        let standard = rhs.t().as_standard_layout().into_owned();
        check_product(view, rhs.t());
        check_product(view, standard.view());
    }
}

fn check_product<T: CubeElement + ndarray::NdFloat>(
    lhs: ArrayView2<'_, T>,
    rhs: ArrayView2<'_, T>,
) {
    let expected = lhs.dot(&rhs);
    for strategy in strategies() {
        let (out, _) = multiply(
            gpu(),
            &upload(gpu(), lhs).unwrap(),
            &upload(gpu(), rhs).unwrap(),
            &strategy,
        )
        .unwrap();
        assert_eq!(download::<T>(gpu(), &out).unwrap(), expected);
    }
}

#[test]
fn zero_dimensions_match_ndarray_without_panicking() {
    for (m, k, n) in [(0, 3, 2), (2, 0, 3), (2, 3, 0)] {
        let a = Array2::<f64>::zeros((m, k));
        let b = Array2::<f64>::zeros((k, n));
        for strategy in strategies() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                multiply(
                    gpu(),
                    &upload(gpu(), a.view()).unwrap(),
                    &upload(gpu(), b.view()).unwrap(),
                    &strategy,
                )
            }));
            assert!(result.is_ok(), "{m}x{k}x{n} {strategy}: setup panicked");
            let (out, _) = result.unwrap().unwrap();
            println!("zero case {m}x{k}x{n} {strategy}: {out:?}");
            assert_eq!(download::<f64>(gpu(), &out).unwrap(), a.dot(&b));
            let a32 = a.mapv(|x| x as f32);
            let b32 = b.mapv(|x| x as f32);
            let (out, _) = multiply(
                gpu(),
                &upload(gpu(), a32.view()).unwrap(),
                &upload(gpu(), b32.view()).unwrap(),
                &strategy,
            )
            .unwrap();
            assert_eq!(download::<f32>(gpu(), &out).unwrap(), a32.dot(&b32));
        }
    }
}

#[test]
fn special_values_match_ndarray() {
    let mut differences = check_specials(&[
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (f64::NEG_INFINITY, 1.0),
        (-0.0, 1.0),
        (f64::MAX, 1.0),
        (f64::from_bits(1), 1.0),
        (f64::MIN_POSITIVE, 0.5),
        (f64::MAX, 2.0),
    ]);
    differences.extend(check_specials(&[
        (f32::NAN, 1.0),
        (f32::INFINITY, 1.0),
        (f32::NEG_INFINITY, 1.0),
        (-0.0, 1.0),
        (f32::MAX, 1.0),
        (f32::from_bits(1), 1.0),
        (f32::MIN_POSITIVE, 0.5),
        (f32::MAX, 2.0),
    ]));
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

fn check_specials<T: CubeElement + ndarray::NdFloat>(values: &[(T, T)]) -> Vec<String> {
    let mut differences = Vec::new();
    for &(value, factor) in values {
        let a = Array2::from_elem((1, 1), value);
        let b = Array2::from_elem((1, 1), factor);
        let expected = a.dot(&b)[[0, 0]];
        for strategy in strategies() {
            let (out, _) = multiply(
                gpu(),
                &upload(gpu(), a.view()).unwrap(),
                &upload(gpu(), b.view()).unwrap(),
                &strategy,
            )
            .unwrap();
            let actual = download::<T>(gpu(), &out).unwrap()[[0, 0]];
            let equal = if expected.is_nan() {
                actual.is_nan()
            } else {
                T::as_bytes(&[actual]) == T::as_bytes(&[expected])
            };
            if !equal {
                differences.push(format!(
                    "{} {value:?} * {factor:?} {strategy}: {actual:?} != {expected:?}",
                    std::any::type_name::<T>()
                ));
            }
        }
    }
    differences
}
