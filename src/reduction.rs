//! GPU-resident reductions with matching input, accumulator and output dtype.
//!
//! Supports f32/f64 on the input device, including strided views. `Some(axis)`
//! removes that axis; `None` reduces to a rank-zero tensor. No result is read
//! back until the caller explicitly downloads it. Summation order and GPU
//! division rounding can differ from CPU results.

use crate::{
    gpu::{Gpu, Result},
    layout::count_elements,
    reduce_kernel,
};
use cubecl::std::tensor::TensorHandle;
use cubek_reduce::components::instructions::ReduceOperationConfig;

/// Sums elements, propagating NaN. An empty reduction yields zero.
/// Returns an error for an invalid axis, unsupported dtype, or another device.
///
/// ```
/// use cubecl::wgpu::WgpuDeviceKind;
/// use ndarray::array;
/// use nu_plugin_vulkan_matrix::{gpu, reduction};
/// # fn main() -> gpu::Result<()> {
/// let gpu = gpu::open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = gpu::upload_array(&gpu, array![[1.0f64, 2.0], [3.0, 4.0]].into_dyn())?;
/// let rows = reduction::sum(&gpu, &input, Some(1))?;
/// let total = reduction::sum(&gpu, &rows, None)?;
/// assert_eq!(gpu::download_array::<f64>(&gpu, &total)?[[]], 10.0);
/// # Ok(())
/// # }
/// ```
pub fn sum(gpu: &Gpu, input: &TensorHandle, axis: Option<usize>) -> Result<TensorHandle> {
    reduce(gpu, input, axis, ReduceOperationConfig::Sum)
}

/// Maximum ignoring NaN, like native matrix. Empty/all-NaN reductions yield -inf.
/// Returns an error for an invalid axis, unsupported dtype, or another device.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use ndarray::array;
/// # use nu_plugin_vulkan_matrix::{gpu, reduction};
/// # fn main() -> gpu::Result<()> {
/// # let gpu = gpu::open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = gpu::upload_array(&gpu, array![f64::NAN, 3.0].into_dyn())?;
/// let maximum = reduction::max(&gpu, &input, None)?;
/// assert_eq!(gpu::download_array::<f64>(&gpu, &maximum)?[[]], 3.0);
/// # Ok(())
/// # }
/// ```
pub fn max(gpu: &Gpu, input: &TensorHandle, axis: Option<usize>) -> Result<TensorHandle> {
    reduce(gpu, input, axis, ReduceOperationConfig::Max)
}

/// Whole-array mean, composed from GPU sum and division in the input dtype.
/// Returns an error for empty input, unsupported dtype, or another device.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use ndarray::array;
/// # use nu_plugin_vulkan_matrix::{gpu, reduction};
/// # fn main() -> gpu::Result<()> {
/// # let gpu = gpu::open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let input = gpu::upload_array(&gpu, array![1.0f64, 3.0].into_dyn())?;
/// let average = reduction::mean(&gpu, &input)?;
/// assert_eq!(gpu::download_array::<f64>(&gpu, &average)?[[]], 2.0);
/// # Ok(())
/// # }
/// ```
pub fn mean(gpu: &Gpu, input: &TensorHandle) -> Result<TensorHandle> {
    use crate::elementwise::{Arithmetic, Operand, apply};
    let n = count_elements(input.shape())?;
    if n == 0 {
        return Err("cannot compute mean of an empty matrix".into());
    }
    let total = sum(gpu, input, None)?;
    apply(
        gpu,
        &total,
        Operand::Scalar(n as f64),
        Arithmetic::Divide,
        false,
    )
}

fn reduce(
    gpu: &Gpu,
    input: &TensorHandle,
    axis: Option<usize>,
    operation: ReduceOperationConfig,
) -> Result<TensorHandle> {
    if input.handle.service != gpu.client.service_id() {
        return Err("matrix belongs to another device".into());
    }
    crate::gpu::check_dtype(gpu, input.dtype)?;
    if let Some(axis) = axis {
        if axis >= input.shape().len() {
            return Err(format!(
                "axis {axis} is out of bounds for a {}-dimensional array",
                input.shape().len()
            )
            .into());
        }
        return reduce_axis(gpu, input, axis, operation);
    }
    let mut output = input.clone();
    if output.shape().is_empty() {
        output = TensorHandle::new_contiguous(vec![1], output.handle, output.dtype);
    }
    while !output.shape().is_empty() {
        let axis = output
            .strides()
            .iter()
            .enumerate()
            .min_by_key(|(_, stride)| **stride)
            .unwrap()
            .0;
        output = reduce_axis(gpu, &output, axis, operation)?;
    }
    Ok(output)
}

fn reduce_axis(
    gpu: &Gpu,
    input: &TensorHandle,
    axis: usize,
    operation: ReduceOperationConfig,
) -> Result<TensorHandle> {
    let mut shape = input.shape().to_vec();
    shape[axis] = 1;
    let size = count_elements(&shape)?;
    let bytes = size
        .checked_mul(input.dtype.size())
        .ok_or("byte size overflow")?;
    let mut output = TensorHandle::new_contiguous(shape, gpu.client.empty(bytes), input.dtype);
    if size != 0 {
        if input.shape()[axis] == 0 {
            let identity = if matches!(operation, ReduceOperationConfig::Max) {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            reduce_kernel::launch_fill(gpu, &output, identity);
        } else {
            reduce_kernel::launch(gpu, input, &output, axis, operation)?;
        }
    }
    output.metadata.shape.remove(axis);
    output.metadata.strides.remove(axis);
    Ok(output)
}
