//! Shape changes share GPU storage and never download elements.
use crate::gpu::Result;
use cubecl::std::tensor::TensorHandle;

/// Reverses all axes, sharing the input buffer.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::{gpu, layout};
/// # let gpu = gpu::open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// # let a = ndarray::array![[1.0_f64, 2.0]];
/// let tensor = gpu::upload(&gpu, a.view())?;
/// assert_eq!(layout::transpose(&tensor).shape().as_slice(), &[2, 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn transpose(input: &TensorHandle) -> TensorHandle {
    let mut output = input.clone();
    output.metadata.shape.reverse();
    output.metadata.strides.reverse();
    output
}

/// Reshapes row-major contiguous storage without copying, like native Nu matrix reshape.
/// Nonstandard layouts and mismatched element counts are errors.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::{gpu, layout};
/// # let gpu = gpu::open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// # let a = ndarray::array![[1.0_f64, 2.0]];
/// let tensor = gpu::upload(&gpu, a.view())?;
/// assert_eq!(layout::reshape(&tensor, &[2])?.shape().as_slice(), &[2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn reshape(input: &TensorHandle, shape: &[usize]) -> Result<TensorHandle> {
    if input.shape().len() != input.strides().len() {
        return Err("shape and stride ranks differ".into());
    }
    let size = count_elements(shape)?;
    if size != count_elements(input.shape())? {
        return Err("cannot reshape: element counts differ".into());
    }
    let mut stride = 1;
    if size != 0 {
        for (&n, &s) in input.shape().iter().zip(input.strides().iter()).rev() {
            if n > 1 && s != stride {
                return Err("cannot reshape a nonstandard layout".into());
            }
            stride *= n;
        }
    }
    Ok(TensorHandle::new_contiguous(
        shape.to_vec(),
        input.handle.clone(),
        input.dtype,
    ))
}

pub(crate) fn count_elements(shape: &[usize]) -> Result<usize> {
    let size = shape
        .iter()
        .filter(|&&n| n != 0)
        .try_fold(1usize, |size, &n| {
            size.checked_mul(n).ok_or("shape size overflow")
        })?;
    if size > isize::MAX as usize {
        return Err("shape size overflow".into());
    }
    Ok(if shape.contains(&0) { 0 } else { size })
}
