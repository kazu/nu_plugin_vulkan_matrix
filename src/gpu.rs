use cubecl::{CubeElement, bytes::Bytes, client::Client, std::tensor::TensorHandle};
use cubecl_runtime::runtime::Runtime;
use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, IxDyn, ShapeBuilder};
use vulkan_matrix_backend::{
    AutoCompiler, RuntimeOptions, Vulkan, WgpuBackend, WgpuDevice, WgpuDeviceKind, WgpuRuntime,
    try_init_setup, wgpu,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone)]
pub struct Gpu {
    pub client: Client,
    pub(crate) device: wgpu::Device,
    pub name: String,
    pub f64: bool,
    pub denorm32: bool,
    pub denorm64: bool,
}

/// Opens an explicit hardware GPU on Vulkan with SPIR-V compilation.
///
/// ```
/// use cubecl::wgpu::WgpuDeviceKind;
/// let gpu = nu_plugin_vulkan_matrix::gpu::open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// assert_eq!(gpu.client.name(), "wgpu<spirv>");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn open_device(kind: WgpuDeviceKind) -> Result<Gpu> {
    if !matches!(
        kind,
        WgpuDeviceKind::DiscreteGpu(_) | WgpuDeviceKind::IntegratedGpu(_)
    ) {
        return Err("select a discrete or integrated GPU explicitly".into());
    }
    let device = WgpuDevice::new(kind).on(WgpuBackend::Vulkan);
    let setup = try_init_setup::<Vulkan>(&device, RuntimeOptions::try_default()?)?;
    let client = WgpuRuntime::<AutoCompiler>::client(&device);
    if client.name() != "wgpu<spirv>" {
        return Err(format!("unexpected compiler: {}", client.name()).into());
    }
    let mut controls = ash::vk::PhysicalDeviceFloatControlsProperties::default();
    let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut controls);
    // The guard keeps the adapter alive; this only queries immutable device properties.
    unsafe {
        let adapter = setup
            .adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or("Vulkan adapter is unavailable")?;
        adapter
            .shared_instance()
            .raw_instance()
            .get_physical_device_properties2(adapter.raw_physical_device(), &mut properties);
    }
    Ok(Gpu {
        client,
        name: setup.adapter.get_info().name,
        f64: setup.device.features().contains(wgpu::Features::SHADER_F64),
        denorm32: controls.shader_denorm_preserve_float32 == ash::vk::TRUE,
        denorm64: controls.shader_denorm_preserve_float64 == ash::vk::TRUE,
        device: setup.device,
    })
}

/// Uploads a two-dimensional view without retaining the source.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::gpu::{open_device, upload};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = ndarray::array![[1.0_f64, 2.0]];
/// let tensor = upload(&gpu, input.view())?;
/// assert_eq!(tensor.shape().as_slice(), &[1, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn upload<T: CubeElement>(gpu: &Gpu, view: ArrayView2<'_, T>) -> Result<TensorHandle> {
    upload_nd(gpu, view.into_dyn())
}

fn upload_nd<T: CubeElement>(gpu: &Gpu, view: ArrayViewD<'_, T>) -> Result<TensorHandle> {
    check_f64::<T>(gpu)?;
    let (data, strides) = if let Some(slice) = view.as_slice_memory_order()
        && view.strides().iter().all(|&s| s >= 0)
    {
        (
            slice.to_vec(),
            view.strides()
                .iter()
                .map(|&s| s as usize)
                .collect::<Vec<_>>(),
        )
    } else {
        let mut strides = view
            .shape()
            .iter()
            .rev()
            .scan(1, |stride, &n| {
                let result = *stride;
                *stride *= n;
                Some(result)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        (view.iter().copied().collect(), strides)
    };
    Ok(create_tensor::<T>(
        gpu,
        Bytes::from_elems(data),
        view.shape().to_vec(),
        strides,
    ))
}

fn check_f64<T: CubeElement>(gpu: &Gpu) -> Result<()> {
    if T::cube_type() == <f64 as CubeElement>::cube_type() && !gpu.f64 {
        return Err("device does not support f64".into());
    }
    Ok(())
}

fn create_tensor<T: CubeElement>(
    gpu: &Gpu,
    data: Bytes,
    shape: Vec<usize>,
    strides: Vec<usize>,
) -> TensorHandle {
    let handle = gpu.client.create(data);
    TensorHandle::new(handle, shape, strides, T::cube_type())
}

/// Reads a two-dimensional tensor with a matching element type.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::gpu::{open_device, upload, download};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = ndarray::array![[1.0_f64, 2.0]];
/// let tensor = upload(&gpu, input.view())?;
/// assert_eq!(download::<f64>(&gpu, &tensor)?, input);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn download<T: CubeElement>(gpu: &Gpu, tensor: &TensorHandle) -> Result<Array2<T>> {
    if tensor.dtype != T::cube_type() || tensor.shape().len() != 2 {
        return Err("download requires matching dtype and rank two".into());
    }
    Ok(read_array(gpu, tensor)?.into_dimensionality()?)
}

fn read_array<T: CubeElement>(gpu: &Gpu, tensor: &TensorHandle) -> Result<ArrayD<T>> {
    let bytes = gpu.client.read_one(tensor.handle.clone())?;
    let data = bytes
        .try_into_vec::<T>()
        .unwrap_or_else(|b| T::from_bytes(&b).to_vec());
    let shape = IxDyn(tensor.shape());
    if data.is_empty() {
        Ok(ArrayD::from_shape_vec(shape, data)?)
    } else {
        Ok(ArrayD::from_shape_vec(
            shape.strides(IxDyn(tensor.strides())),
            data,
        )?)
    }
}

/// Uploads any-rank f32/f64 view, retaining positive contiguous strides or packing once.
/// Unsupported precision is rejected; the source can be dropped after this call.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::gpu::{open_device, upload_view};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 3, 4]), 1.0_f64);
/// let tensor = upload_view(&gpu, input.view())?;
/// assert_eq!(tensor.shape().as_slice(), &[2, 3, 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn upload_view<T: CubeElement>(gpu: &Gpu, view: ArrayViewD<'_, T>) -> Result<TensorHandle> {
    check_precision::<T>(gpu)?;
    upload_nd(gpu, view)
}

/// Uploads an owned f32/f64 array. Positive contiguous storage moves without a host data copy;
/// offset windows share that allocation, and noncontiguous/negative strides pack once.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::gpu::{open_device, upload_array};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0_f32);
/// let tensor = upload_array(&gpu, input)?;
/// assert_eq!(tensor.shape().as_slice(), &[4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn upload_array<T: CubeElement>(gpu: &Gpu, array: ArrayD<T>) -> Result<TensorHandle> {
    check_precision::<T>(gpu)?;
    if array.as_slice_memory_order().is_none() || array.strides().iter().any(|&s| s < 0) {
        return upload_nd(gpu, array.view());
    }
    let shape = array.shape().to_vec();
    let strides = array.strides().iter().map(|&s| s as usize).collect();
    let len = array.len() * size_of::<T>();
    let (data, offset) = array.into_raw_vec_and_offset();
    let offset = offset.unwrap_or(0) * size_of::<T>();
    let mut bytes = Bytes::from_elems(data);
    if bytes.len() != len {
        bytes = bytes
            .shared()
            .view(offset, offset + len)
            .map_err(|error| format!("invalid array byte range: {error:?}"))?;
    }
    Ok(create_tensor::<T>(gpu, bytes, shape, strides))
}

/// Reads an f32/f64 tensor of matching dtype into an array with its original shape and strides.
/// Device mismatches and deferred transfer/launch failures remain errors.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::gpu::{open_device, upload_array, download_array};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 3, 4]), 2.0_f64);
/// let tensor = upload_array(&gpu, input.clone())?;
/// assert_eq!(download_array::<f64>(&gpu, &tensor)?, input);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn download_array<T: CubeElement>(gpu: &Gpu, tensor: &TensorHandle) -> Result<ArrayD<T>> {
    check_precision::<T>(gpu)?;
    if tensor.dtype != T::cube_type() {
        return Err("download requires matching dtype".into());
    }
    read_array(gpu, tensor)
}

pub(crate) fn check_precision<T: CubeElement>(gpu: &Gpu) -> Result<()> {
    let preserved = if T::cube_type() == <f32 as CubeElement>::cube_type() {
        gpu.denorm32
    } else if T::cube_type() == <f64 as CubeElement>::cube_type() {
        gpu.denorm64
    } else {
        return Err("dtype must be f32 or f64".into());
    };
    if !preserved {
        return Err("device does not preserve denormals for this dtype".into());
    }
    check_f64::<T>(gpu)
}

pub(crate) fn check_dtype(gpu: &Gpu, dtype: cubecl::ir::ElemType) -> Result<()> {
    if dtype == f64::cube_type() {
        check_precision::<f64>(gpu)
    } else if dtype == f32::cube_type() {
        check_precision::<f32>(gpu)
    } else {
        Err("dtype must be f32 or f64".into())
    }
}
