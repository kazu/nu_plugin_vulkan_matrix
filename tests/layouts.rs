use cubecl::wgpu::WgpuDeviceKind;
use ndarray::{ArrayD, IxDyn, s};
use nu_plugin_vulkan_matrix::{
    gpu::{download_array, open_device, upload_view},
    layout,
};

fn gpu() -> &'static nu_plugin_vulkan_matrix::gpu::Gpu {
    static GPU: std::sync::OnceLock<nu_plugin_vulkan_matrix::gpu::Gpu> = std::sync::OnceLock::new();
    GPU.get_or_init(|| open_device(WgpuDeviceKind::DiscreteGpu(0)).unwrap())
}

#[test]
fn rejects_invalid_empty_shapes_and_incomplete_strides() {
    let gpu = gpu();
    let empty = upload_view(gpu, ArrayD::<f64>::zeros(IxDyn(&[0])).view()).unwrap();
    for shape in [[0, usize::MAX, 2], [0, isize::MAX as usize + 1, 1]] {
        assert!(layout::reshape(&empty, &shape).is_err());
    }
    let mut tensor = upload_view(gpu, ndarray::array![[1.0_f64, 2.0]].into_dyn().view()).unwrap();
    tensor.metadata.strides.remove(1);
    assert!(layout::reshape(&tensor, &[2]).is_err());
}

#[test]
fn layouts_match_ndarray_without_changing_the_input() {
    let gpu = gpu();
    for shape in [&[][..], &[6], &[2, 3], &[2, 1, 3], &[2, 0, 3]] {
        let a = ArrayD::from_shape_fn(IxDyn(shape), |idx| {
            use ndarray::Dimension;
            idx.slice().iter().fold(0., |v, &i| v * 10. + i as f64)
        });
        let tensor = upload_view(gpu, a.view()).unwrap();
        let transposed = layout::transpose(&tensor);
        assert_eq!(
            transposed.handle.memory.descriptor().id,
            tensor.handle.memory.descriptor().id
        );
        assert_eq!(
            download_array::<f64>(gpu, &transposed).unwrap(),
            a.clone().reversed_axes()
        );
        let flat = layout::reshape(&tensor, &[a.len()]).unwrap();
        assert_eq!(
            flat.handle.memory.descriptor().id,
            tensor.handle.memory.descriptor().id
        );
        assert_eq!(
            download_array::<f64>(gpu, &flat).unwrap(),
            a.clone().into_shape_with_order(IxDyn(&[a.len()])).unwrap()
        );
        let expected = a
            .clone()
            .reversed_axes()
            .into_shape_with_order(IxDyn(&[a.len()]));
        assert_eq!(
            layout::reshape(&transposed, &[a.len()]).is_ok(),
            expected.is_ok()
        );
        assert_eq!(download_array::<f64>(gpu, &tensor).unwrap(), a);
        assert!(layout::reshape(&tensor, &[a.len() + 1]).is_err());
    }
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|n| n as f32).collect()).unwrap();
    let negative = a.slice(s![.., ..;-1, ..;2]).into_dyn();
    let tensor = upload_view(gpu, negative.clone()).unwrap();
    assert_eq!(
        download_array::<f32>(gpu, &layout::transpose(&tensor)).unwrap(),
        negative.reversed_axes()
    );
    assert!(layout::reshape(&tensor, &[usize::MAX, 2]).is_err());
}
