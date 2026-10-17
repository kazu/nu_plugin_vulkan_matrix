use cubecl::{std::tensor::TensorHandle, wgpu::WgpuDeviceKind};
use ndarray::{ArrayD, IxDyn, s};
use nu_plugin_vulkan_matrix::gpu::{Gpu, download_array, open_device, upload_array, upload_view};
use std::sync::OnceLock;

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        let kind = match std::env::var("VM_GPU").as_deref().unwrap_or("d0") {
            "d0" => WgpuDeviceKind::DiscreteGpu(0),
            "d1" => WgpuDeviceKind::DiscreteGpu(1),
            "i0" => WgpuDeviceKind::IntegratedGpu(0),
            other => panic!("unknown selector {other}"),
        };
        open_device(kind).unwrap()
    })
}

fn check_f64(tensor: &TensorHandle, expected: &ArrayD<f64>) {
    let actual = download_array::<f64>(gpu(), tensor).unwrap();
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn strided_nd_views_and_owned_arrays_keep_shape_and_values() {
    let input =
        ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), (0..60).map(|n| n as f64).collect()).unwrap();
    for view in [
        input.view(),
        input.view().permuted_axes(IxDyn(&[2, 0, 1])),
        input.slice(s![..;-1, 1..4;2, ..;2]).into_dyn(),
    ] {
        check_f64(&upload_view(gpu(), view.clone()).unwrap(), &view.to_owned());
    }
    for owned in [
        input.clone(),
        input.clone().permuted_axes(IxDyn(&[2, 0, 1])),
        input.clone().slice_move(s![1..3, .., ..]).into_dyn(),
        input.slice_move(s![..;-1, .., ..;2]).into_dyn(),
    ] {
        let expected = owned.clone();
        let tensor = upload_array(gpu(), owned).unwrap();
        check_f64(&tensor, &expected);
    }
}

#[test]
fn scalar_vector_and_empty_axes_round_trip() {
    for shape in [&[][..], &[4], &[0, 2, 3], &[2, 0, 3], &[2, 3, 0]] {
        let input = ArrayD::from_elem(IxDyn(shape), -0.0_f64);
        check_f64(&upload_view(gpu(), input.view()).unwrap(), &input);
        check_f64(&upload_array(gpu(), input.clone()).unwrap(), &input);
    }
}

#[test]
fn f32_bits_are_preserved_and_mismatched_types_are_rejected() {
    let values = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x7f800000, 0xff800000, 0x7fc01234,
    ];
    let input =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), values.map(f32::from_bits).to_vec()).unwrap();
    let tensor = upload_array(gpu(), input).unwrap();
    assert_eq!(
        download_array::<f32>(gpu(), &tensor)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        values
    );
    assert!(download_array::<f64>(gpu(), &tensor).is_err());
    assert!(upload_array(gpu(), ArrayD::from_elem(IxDyn(&[1]), 1_u32)).is_err());
}

#[test]
fn unsupported_precision_is_rejected_before_transfer() {
    let mut limited = gpu().clone();
    limited.f64 = false;
    limited.denorm32 = true;
    limited.denorm64 = true;
    let doubles = ArrayD::from_elem(IxDyn(&[1]), 1.0_f64);
    assert!(upload_view(&limited, doubles.view()).is_err());
    assert!(upload_array(&limited, doubles).is_err());
    limited.f64 = true;
    limited.denorm64 = false;
    assert!(upload_array(&limited, ArrayD::from_elem(IxDyn(&[1]), 1.0_f64)).is_err());
    limited.denorm32 = false;
    assert!(upload_array(&limited, ArrayD::from_elem(IxDyn(&[1]), 1.0_f32)).is_err());
    let tensor = upload_array(gpu(), ArrayD::from_elem(IxDyn(&[1]), 1.0_f32)).unwrap();
    assert!(download_array::<f32>(&limited, &tensor).is_err());
}

#[test]
fn a_different_device_cannot_read_the_tensor() {
    let other_kind = if std::env::var("VM_GPU").as_deref() == Ok("i0") {
        WgpuDeviceKind::DiscreteGpu(0)
    } else {
        WgpuDeviceKind::IntegratedGpu(0)
    };
    let other = open_device(other_kind).unwrap();
    let tensor = upload_array(gpu(), ArrayD::from_elem(IxDyn(&[3]), 1.0_f64)).unwrap();
    assert!(download_array::<f64>(&other, &tensor).is_err());
}
