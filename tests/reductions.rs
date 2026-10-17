use cubecl::wgpu::WgpuDeviceKind;
use ndarray::{ArrayD, Axis, IxDyn};
use nu_plugin_vulkan_matrix::{gpu, layout, reduction};

fn check_sum(actual: f64, values: impl Iterator<Item = f64>, unit: f64) {
    let (expected, magnitude, count) = values.fold((0.0, 0.0, 0usize), |(sum, magnitude, n), x| {
        (sum + x, magnitude + x.abs(), n + 1)
    });
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else if expected.is_infinite() {
        assert_eq!(actual, expected);
    } else {
        let n = count.max(1) as f64;
        let gamma = n * unit / (1.0 - n * unit);
        let bound = 2.0 * gamma * magnitude;
        assert!(
            (actual - expected).abs() <= bound,
            "{actual} != {expected}, bound {bound}"
        );
    }
}

#[test]
fn reductions_match_ndarray_on_three_devices() {
    for device in [
        WgpuDeviceKind::DiscreteGpu(0),
        WgpuDeviceKind::DiscreteGpu(1),
        WgpuDeviceKind::IntegratedGpu(0),
    ] {
        let gpu = gpu::open_device(device).unwrap();
        macro_rules! check_nan_rows {
            ($t:ty) => {{
                let array = ndarray::array![
                    [<$t>::NAN, <$t>::NAN, <$t>::NAN],
                    [<$t>::NAN, 2.0, -1.0],
                    [1.0, 2.0, 3.0],
                    [<$t>::INFINITY, <$t>::NEG_INFINITY, 3.0],
                ]
                .into_dyn();
                let input = gpu::upload_view(&gpu, array.view()).unwrap();
                let output = reduction::max(&gpu, &input, Some(1)).unwrap();
                assert_eq!(
                    gpu::download_array::<$t>(&gpu, &output).unwrap(),
                    array.map_axis(Axis(1), |v| v
                        .iter()
                        .copied()
                        .fold(<$t>::NEG_INFINITY, <$t>::max))
                );
                let output = reduction::sum(&gpu, &input, Some(1)).unwrap();
                let actual = gpu::download_array::<$t>(&gpu, &output).unwrap();
                for (&a, &b) in actual.iter().zip(array.sum_axis(Axis(1)).iter()) {
                    assert!(if b.is_nan() { a.is_nan() } else { a == b });
                }
            }};
        }
        check_nan_rows!(f64);
        check_nan_rows!(f32);
        for shape in [vec![2, 3, 5], vec![4097], vec![0, 3], vec![2, 0], vec![]] {
            let array = ArrayD::from_shape_fn(IxDyn(&shape), |i| {
                (i.slice().iter().sum::<usize>() % 17) as f64 / 16.0 - 0.5
            });
            let input = gpu::upload_view(&gpu, array.view()).unwrap();
            for transposed in [false, true] {
                let input = if transposed {
                    layout::transpose(&input)
                } else {
                    input.clone()
                };
                let expected = if transposed {
                    array.clone().reversed_axes()
                } else {
                    array.clone()
                };
                for axis in (0..expected.ndim()).map(Some).chain([None]) {
                    let output = reduction::sum(&gpu, &input, axis).unwrap();
                    let actual = gpu::download_array::<f64>(&gpu, &output).unwrap();
                    let maximum = reduction::max(&gpu, &input, axis).unwrap();
                    let maximum = gpu::download_array::<f64>(&gpu, &maximum).unwrap();
                    if let Some(axis) = axis {
                        let sums = expected.sum_axis(Axis(axis));
                        assert_eq!(actual.shape(), sums.shape());
                        for (&value, lane) in actual.iter().zip(expected.lanes(Axis(axis))) {
                            check_sum(value, lane.iter().copied(), f64::EPSILON / 2.0);
                        }
                        assert_eq!(
                            maximum,
                            expected.map_axis(Axis(axis), |v| v
                                .iter()
                                .copied()
                                .fold(f64::NEG_INFINITY, f64::max))
                        );
                    } else {
                        assert!(actual.shape().is_empty());
                        check_sum(
                            actual[IxDyn(&[])],
                            expected.iter().copied(),
                            f64::EPSILON / 2.0,
                        );
                        assert_eq!(
                            maximum[IxDyn(&[])],
                            expected.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                        );
                    }
                }
                if expected.is_empty() {
                    assert!(reduction::mean(&gpu, &input).is_err());
                } else {
                    let mean = reduction::mean(&gpu, &input).unwrap();
                    let mean = gpu::download_array::<f64>(&gpu, &mean).unwrap()[IxDyn(&[])];
                    let reference = expected.sum() / expected.len() as f64;
                    assert!(
                        (mean - reference).abs() <= 4.0 * f64::EPSILON * reference.abs().max(1.0)
                    );
                }
                assert!(reduction::sum(&gpu, &input, Some(expected.ndim())).is_err());
                assert!(reduction::max(&gpu, &input, Some(expected.ndim())).is_err());
            }
        }
        for values in [
            vec![f64::NAN, 2.0, -1.0],
            vec![f64::NAN; 3],
            vec![f64::NEG_INFINITY; 3],
            vec![f64::INFINITY, f64::NEG_INFINITY],
            vec![-0.0, 0.0],
            vec![1e16, 1.0, -1e16, 1.0],
        ] {
            let a = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.clone()).unwrap();
            let a = gpu::upload_array(&gpu, a).unwrap();
            let mean = reduction::mean(&gpu, &a).unwrap();
            check_sum(
                gpu::download_array::<f64>(&gpu, &mean).unwrap()[IxDyn(&[])] * values.len() as f64,
                values.iter().copied(),
                f64::EPSILON / 2.0,
            );
            for axis in [None, Some(0)] {
                let sum = reduction::sum(&gpu, &a, axis).unwrap();
                check_sum(
                    gpu::download_array::<f64>(&gpu, &sum).unwrap()[IxDyn(&[])],
                    values.iter().copied(),
                    f64::EPSILON / 2.0,
                );
                let max = reduction::max(&gpu, &a, axis).unwrap();
                assert_eq!(
                    gpu::download_array::<f64>(&gpu, &max).unwrap()[IxDyn(&[])],
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                );
            }
            let a32 = ArrayD::from_shape_vec(
                IxDyn(&[values.len()]),
                values.iter().map(|&v| v as f32).collect(),
            )
            .unwrap();
            let a32 = gpu::upload_array(&gpu, a32).unwrap();
            let mean = reduction::mean(&gpu, &a32).unwrap();
            check_sum(
                gpu::download_array::<f32>(&gpu, &mean).unwrap()[IxDyn(&[])] as f64
                    * values.len() as f64,
                values.iter().map(|&v| (v as f32) as f64),
                f32::EPSILON as f64 / 2.0,
            );
            for axis in [None, Some(0)] {
                let max = reduction::max(&gpu, &a32, axis).unwrap();
                assert_eq!(
                    gpu::download_array::<f32>(&gpu, &max).unwrap()[IxDyn(&[])],
                    values
                        .iter()
                        .map(|&v| v as f32)
                        .fold(f32::NEG_INFINITY, f32::max)
                );
                let sum = reduction::sum(&gpu, &a32, axis).unwrap();
                check_sum(
                    gpu::download_array::<f32>(&gpu, &sum).unwrap()[IxDyn(&[])] as f64,
                    values.iter().map(|&v| (v as f32) as f64),
                    f32::EPSILON as f64 / 2.0,
                );
            }
        }
    }
}

use ndarray::Dimension;
