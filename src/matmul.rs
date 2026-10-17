use crate::gpu::{Gpu, Result};
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    launch::launch_ref,
    strategy::Strategy,
};
use cubek_std::InputBinding;

/// Multiplies matching 1D/2D f32/f64 arrays with SimpleUnit, preserving internal precision.
/// Shape, device, dtype, buffer bounds and output allocation limits are checked before launch.
/// The returned tensor is pending; downloading or synchronizing it reports deferred failures.
/// Two vectors produce a rank-zero tensor; a single vector removes its corresponding output axis.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::{gpu::{open_device, upload, download}, matmul::multiply};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let a = ndarray::array![[1.0_f64, 2.0]];
/// let b = ndarray::array![[3.0_f64], [4.0]];
/// let out = multiply(&gpu, &upload(&gpu, a.view())?, &upload(&gpu, b.view())?)?;
/// assert_eq!(download::<f64>(&gpu, &out)?, a.dot(&b));
/// let vector = nu_plugin_vulkan_matrix::gpu::upload_array(&gpu, ndarray::array![1.0_f64, 2.0].into_dyn())?;
/// let scalar = multiply(&gpu, &vector, &vector)?;
/// assert!(scalar.shape().is_empty());
/// assert_eq!(nu_plugin_vulkan_matrix::gpu::download_array::<f64>(&gpu, &scalar)?[[]], 5.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn multiply(gpu: &Gpu, lhs: &TensorHandle, rhs: &TensorHandle) -> Result<TensorHandle> {
    if lhs.dtype != rhs.dtype {
        return Err("multiply requires matching dtypes".into());
    }
    crate::gpu::check_dtype(gpu, lhs.dtype)?;
    for tensor in [lhs, rhs] {
        if tensor.handle.service != gpu.client.service_id() {
            return Err("matrix belongs to another device".into());
        }
        if !(1..=2).contains(&tensor.shape().len())
            || tensor.strides().len() != tensor.shape().len()
        {
            return Err("multiply only supports 1D and 2D arrays".into());
        }
        let elements = if tensor.shape().contains(&0) {
            0
        } else {
            tensor
                .shape()
                .iter()
                .zip(tensor.strides().iter())
                .try_fold(1usize, |end, (&n, &s)| {
                    end.checked_add((n - 1).checked_mul(s)?)
                })
                .ok_or("input buffer range overflow")?
        };
        let bytes = elements
            .checked_mul(tensor.dtype.size())
            .ok_or("input byte size overflow")?;
        if bytes as u64 > tensor.handle.size_in_used() {
            return Err("matrix shape and strides exceed its buffer".into());
        }
    }
    let lhs_vector = lhs.shape().len() == 1;
    let rhs_vector = rhs.shape().len() == 1;
    let lhs_view;
    let lhs = if lhs_vector {
        lhs_view = {
            let mut tensor = lhs.clone();
            tensor.metadata.shape.insert(0, 1);
            tensor.metadata.strides.insert(0, 1);
            tensor
        };
        &lhs_view
    } else {
        lhs
    };
    let rhs_view;
    let rhs = if rhs_vector {
        rhs_view = {
            let mut tensor = rhs.clone();
            tensor.metadata.shape.insert(1, 1);
            tensor.metadata.strides.insert(1, 1);
            tensor
        };
        &rhs_view
    } else {
        rhs
    };
    if lhs.shape()[1] != rhs.shape()[0] {
        return Err("inner dimensions do not match".into());
    }
    let bytes = lhs.shape()[0]
        .checked_mul(rhs.shape()[1])
        .and_then(|n| n.checked_mul(lhs.dtype.size()))
        .ok_or("output size overflow")?;
    if bytes as u64 > gpu.client.properties().memory.max_page_size {
        return Err("output exceeds the device allocation limit".into());
    }
    let (mut out, _) = launch_multiply(
        gpu,
        lhs,
        rhs,
        &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::SimpleUnit(
            Default::default(),
        )),
    )?;
    if rhs_vector {
        out.metadata.shape.remove(1);
        out.metadata.strides.remove(1);
    }
    if lhs_vector {
        out.metadata.shape.remove(0);
        out.metadata.strides.remove(0);
    }
    Ok(out)
}

/// Multiplies compatible 2D tensors using exactly the requested CubeK strategy.
/// Returns the selected internal types, rejecting an implicit precision change.
/// CubeK's untiled `Auto` probe can panic on setup errors other than `Unavailable`;
/// this experiment does not turn that upstream panic branch into a fallible API.
///
/// ```
/// # use cubecl::wgpu::WgpuDeviceKind;
/// # use nu_plugin_vulkan_matrix::{gpu::{open_device, upload, download}, matmul::launch_multiply};
/// # let gpu = open_device(WgpuDeviceKind::DiscreteGpu(0))?;
/// let source = ndarray::array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let a = upload(&gpu, source.view())?;
/// let (out, types) = launch_multiply(&gpu, &a, &a, &cubek_matmul::strategy::Strategy::Auto)?;
/// assert_eq!(download::<f64>(&gpu, &out)?, source.dot(&source));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn launch_multiply(
    gpu: &Gpu,
    lhs: &TensorHandle,
    rhs: &TensorHandle,
    strategy: &Strategy,
) -> Result<(TensorHandle, MatmulElems)> {
    if lhs.shape().len() != 2
        || rhs.shape().len() != 2
        || lhs.shape()[1] != rhs.shape()[0]
        || lhs.dtype != rhs.dtype
    {
        return Err("multiply requires compatible 2D shapes and matching dtypes".into());
    }
    let out = TensorHandle::empty(&gpu.client, vec![lhs.shape()[0], rhs.shape()[1]], lhs.dtype);
    let mut elems = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs.dtype,
        rhs: rhs.dtype,
        out: lhs.dtype,
    });
    launch_ref(
        strategy,
        &gpu.client,
        InputBinding::Normal(lhs.clone().binding(), lhs.dtype),
        InputBinding::Normal(rhs.clone().binding(), rhs.dtype),
        out.clone().binding(),
        &mut elems,
    )
    .map_err(|e| format!("{strategy}: {e:?}"))?;
    if [
        elems.lhs_stage,
        elems.rhs_stage,
        elems.acc_stage,
        elems.lhs_register,
        elems.rhs_register,
        elems.acc_register,
    ]
    .iter()
    .any(|&elem| elem != lhs.dtype)
    {
        return Err(format!("{strategy} changed internal precision: {elems:?}").into());
    }
    Ok((out, elems))
}
