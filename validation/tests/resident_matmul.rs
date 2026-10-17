mod support;

use cubecl::std::tensor::TensorHandle;
use ndarray::{Array2, array};
use nu_plugin_vulkan_matrix::{
    gpu::{download, upload},
    matmul::multiply,
};

#[test]
fn rectangular_transposed_and_precision_match_ndarray() {
    let gpu = support::gpu();
    let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    for view in [a.view(), a.t()] {
        let lhs = upload(gpu, view).unwrap();
        let rhs = upload(gpu, view.t()).unwrap();
        let out = multiply(gpu, &lhs, &rhs).unwrap();
        assert_eq!(download::<f64>(gpu, &out).unwrap(), view.dot(&view.t()));
    }
    let a = array![[1.0_f64 + 2.0_f64.powi(-30)]];
    let lhs = upload(gpu, a.view()).unwrap();
    let rhs = upload(gpu, array![[1.0_f64]].view()).unwrap();
    assert_eq!(
        download::<f64>(gpu, &multiply(gpu, &lhs, &rhs).unwrap()).unwrap(),
        a
    );
}

#[test]
fn malformed_buffers_and_foreign_devices_return_errors() {
    let gpu = support::gpu();
    let a = upload(gpu, array![[1.0_f64]].view()).unwrap();
    let short = TensorHandle::new(a.handle.clone(), vec![2, 2], vec![2, 1], a.dtype);
    assert!(multiply(gpu, &short, &short).is_err());
    let overflow = TensorHandle::new(a.handle.clone(), vec![usize::MAX, 2], vec![2, 1], a.dtype);
    assert!(multiply(gpu, &overflow, &a).is_err());
    let huge_lhs = TensorHandle::new(a.handle.clone(), vec![usize::MAX, 1], vec![0, 0], a.dtype);
    let huge_rhs = TensorHandle::new(a.handle.clone(), vec![1, usize::MAX], vec![0, 0], a.dtype);
    assert!(
        multiply(gpu, &huge_lhs, &huge_rhs)
            .unwrap_err()
            .to_string()
            .contains("output size overflow")
    );
    let n = (gpu.client.properties().memory.max_page_size / 8) as usize + 1;
    let huge_rhs = TensorHandle::new(a.handle.clone(), vec![1, n], vec![0, 0], a.dtype);
    assert!(
        multiply(gpu, &a, &huge_rhs)
            .unwrap_err()
            .to_string()
            .contains("allocation limit")
    );
    let invalid = TensorHandle::new(
        a.handle.clone(),
        vec![1, 1],
        vec![1, 1],
        <i32 as cubecl::CubeElement>::cube_type(),
    );
    assert!(
        multiply(gpu, &invalid, &invalid)
            .unwrap_err()
            .to_string()
            .contains("dtype")
    );
    let dtype = upload(gpu, array![[1.0_f32]].view()).unwrap();
    assert!(multiply(gpu, &a, &dtype).is_err());
    let mut foreign = a.clone();
    let other_index = if std::env::var("VM_GPU").as_deref() == Ok("d1") {
        0
    } else {
        1
    };
    let other = nu_plugin_vulkan_matrix::gpu::open_device(
        cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(other_index),
    )
    .unwrap();
    foreign.handle = upload(&other, array![[1.0_f64]].view()).unwrap().handle;
    if other.client.service_id() != gpu.client.service_id() {
        assert!(multiply(gpu, &foreign, &a).is_err());
    }
}

#[test]
fn empty_tile_boundaries_and_specials_match_ndarray() {
    let gpu = support::gpu();
    for (m, k, n) in [(1, 1, 1), (31, 33, 65), (0, 3, 2), (2, 0, 3), (2, 3, 0)] {
        let a = Array2::from_shape_fn((m, k), |(i, j)| ((i * 7 + j) % 5) as f64 - 2.0);
        let b = Array2::from_shape_fn((k, n), |(i, j)| ((i + j * 3) % 5) as f64 - 2.0);
        let expected = a.dot(&b);
        let out = multiply(
            gpu,
            &upload(gpu, a.view()).unwrap(),
            &upload(gpu, b.view()).unwrap(),
        )
        .unwrap();
        assert_eq!(download::<f64>(gpu, &out).unwrap(), expected);
        let a = a.mapv(|v| v as f32);
        let b = b.mapv(|v| v as f32);
        let out = multiply(
            gpu,
            &upload(gpu, a.view()).unwrap(),
            &upload(gpu, b.view()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            download::<f32>(gpu, &out).unwrap(),
            expected.mapv(|v| v as f32)
        );
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = upload(gpu, array![[value]].view()).unwrap();
        let b = upload(gpu, array![[1.0_f64]].view()).unwrap();
        let got = download::<f64>(gpu, &multiply(gpu, &a, &b).unwrap()).unwrap()[[0, 0]];
        let a = upload(gpu, array![[value as f32]].view()).unwrap();
        let b = upload(gpu, array![[1.0_f32]].view()).unwrap();
        let got32 = download::<f32>(gpu, &multiply(gpu, &a, &b).unwrap()).unwrap()[[0, 0]];
        for (dtype_index, got) in [got, f64::from(got32)].into_iter().enumerate() {
            assert!(
                if value.is_nan() {
                    got.is_nan()
                } else {
                    got == value
                },
                "expected={value:?} actual={got:?} dtype_index={dtype_index}"
            );
        }
    }
    let a = upload(gpu, array![[f64::from_bits(1)]].view()).unwrap();
    let b = upload(gpu, array![[1.0_f64]].view()).unwrap();
    assert_eq!(
        download::<f64>(gpu, &multiply(gpu, &a, &b).unwrap()).unwrap()[[0, 0]].to_bits(),
        1
    );
    let a = upload(gpu, array![[f32::from_bits(1)]].view()).unwrap();
    let b = upload(gpu, array![[1.0_f32]].view()).unwrap();
    assert_eq!(
        download::<f32>(gpu, &multiply(gpu, &a, &b).unwrap()).unwrap()[[0, 0]].to_bits(),
        1
    );
}

#[test]
fn roundoff_uses_inner_length_and_absolute_product_sum() {
    let gpu = support::gpu();
    let a = Array2::from_shape_fn((3, 65), |(i, j)| ((i + j * 7) % 19) as f64 / 7.0 - 1.0);
    let b = Array2::from_shape_fn((65, 4), |(i, j)| ((i * 3 + j) % 17) as f64 / 11.0 - 1.0);
    let lhs = upload(gpu, a.view()).unwrap();
    let rhs = upload(gpu, b.view()).unwrap();
    let result = multiply(gpu, &lhs, &rhs).unwrap();
    drop(lhs);
    drop(rhs);
    check_roundoff(
        &download::<f64>(gpu, &result).unwrap(),
        &a,
        &b,
        f64::EPSILON / 2.0,
    );
    let a = a.mapv(|x| x as f32);
    let b = b.mapv(|x| x as f32);
    let result = multiply(
        gpu,
        &upload(gpu, a.view()).unwrap(),
        &upload(gpu, b.view()).unwrap(),
    )
    .unwrap();
    check_roundoff(
        &download::<f32>(gpu, &result).unwrap().mapv(f64::from),
        &a.mapv(f64::from),
        &b.mapv(f64::from),
        f32::EPSILON as f64 / 2.0,
    );
    let a = array![[1.0_f64, 1.0, 1.0]];
    let b = array![[1.0_f64], [-1.0], [2.0_f64.powi(-50)]];
    let out = multiply(
        gpu,
        &upload(gpu, a.view()).unwrap(),
        &upload(gpu, b.view()).unwrap(),
    )
    .unwrap();
    check_roundoff(
        &download::<f64>(gpu, &out).unwrap(),
        &a,
        &b,
        f64::EPSILON / 2.0,
    );
}

fn check_roundoff(got: &Array2<f64>, a: &Array2<f64>, b: &Array2<f64>, u: f64) {
    let expected = a.dot(b);
    let operations = 2.0 * a.ncols() as f64;
    let gamma = operations * u / (1.0 - operations * u);
    for ((i, j), actual) in got.indexed_iter() {
        let sum_abs = (0..a.ncols())
            .map(|k| (a[[i, k]] * b[[k, j]]).abs())
            .sum::<f64>();
        let bound = 2.0 * gamma / (1.0 - gamma) * sum_abs;
        assert!(
            (actual - expected[[i, j]]).abs() <= bound,
            "({i},{j}): {actual} vs {} bound={bound}",
            expected[[i, j]]
        );
    }
}

#[test]
fn vectors_return_the_ndarray_shapes_and_can_chain_on_gpu() {
    use ndarray::{ArrayD, IxDyn};
    use nu_plugin_vulkan_matrix::gpu::{download_array, upload_array};
    let gpu = support::gpu();
    let cases = [
        (
            array![1.0_f64, 2.0, 3.0].into_dyn(),
            array![4.0_f64, 5.0, 6.0].into_dyn(),
            ArrayD::from_elem(IxDyn(&[]), 32.0),
        ),
        (
            array![1.0_f64, 2.0].into_dyn(),
            array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
            array![9.0, 12.0, 15.0].into_dyn(),
        ),
        (
            array![[1.0_f64, 2.0], [3.0, 4.0]].into_dyn(),
            array![5.0_f64, 6.0].into_dyn(),
            array![17.0, 39.0].into_dyn(),
        ),
        (
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[])),
        ),
        (
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[0, 3])),
            ArrayD::zeros(IxDyn(&[3])),
        ),
        (
            ArrayD::zeros(IxDyn(&[2, 0])),
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[2])),
        ),
        (
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[0, 0])),
            ArrayD::zeros(IxDyn(&[0])),
        ),
    ];
    for (a, b, expected) in cases {
        let lhs = upload_array(gpu, a.clone()).unwrap();
        let rhs = upload_array(gpu, b.clone()).unwrap();
        let result = multiply(gpu, &lhs, &rhs).unwrap();
        drop(lhs);
        drop(rhs);
        assert_eq!(download_array::<f64>(gpu, &result).unwrap(), expected);
        let lhs = upload_array(gpu, a.mapv(|v| v as f32)).unwrap();
        let rhs = upload_array(gpu, b.mapv(|v| v as f32)).unwrap();
        let result = multiply(gpu, &lhs, &rhs).unwrap();
        assert_eq!(
            download_array::<f32>(gpu, &result).unwrap(),
            expected.mapv(|v| v as f32)
        );
    }
    for shape in [vec![], vec![1, 1, 1]] {
        let invalid = upload_array(gpu, ArrayD::from_elem(IxDyn(&shape), 1.0_f64)).unwrap();
        assert!(multiply(gpu, &invalid, &invalid).is_err());
    }
}
