//! Typed element arithmetic uses fresh outputs; inputs remain immutable.
use crate::{
    gpu::{Gpu, Result, check_precision},
    layout::count_elements,
};
use cubecl::{prelude::*, std::tensor::TensorHandle};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Arithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
}

pub(crate) enum Operand<'a> {
    Tensor(&'a TensorHandle),
    Scalar(f64),
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn evaluate<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    out: &mut Tensor<F>,
    scalar: InputScalar,
    #[comptime] scalar_rhs: bool,
    #[comptime] op: u8,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        let mut ai = 0;
        let mut bi = 0;
        for axis in 0..out.rank() {
            let coordinate = pos / out.stride(axis) % out.shape(axis);
            ai += coordinate * a.stride(axis);
            if !scalar_rhs {
                bi += coordinate * b.stride(axis);
            }
        }
        let x = a[ai];
        let y = if scalar_rhs { scalar.get::<F>() } else { b[bi] };
        out[pos] = if op == 0 {
            x + y
        } else if op == 1 {
            x - y
        } else if op == 2 {
            x * y
        } else {
            x / y
        };
    }
}

pub(crate) fn apply(
    gpu: &Gpu,
    a: &TensorHandle,
    rhs: Operand<'_>,
    op: Arithmetic,
    broadcast: bool,
) -> Result<TensorHandle> {
    let (mut b, scalar, scalar_rhs) = match rhs {
        Operand::Tensor(b) => {
            if a.dtype != b.dtype {
                return Err("element operations require matching dtypes".into());
            }
            if b.handle.service != gpu.client.service_id() {
                return Err("matrix belongs to another device".into());
            }
            if !broadcast && a.shape() != b.shape() {
                return Err("matrix shapes differ; use --broadcast for RHS broadcasting".into());
            }
            (b.clone(), 0., false)
        }
        Operand::Scalar(s) => (a.clone(), s, true),
    };
    if a.handle.service != gpu.client.service_id() {
        return Err("matrix belongs to another device".into());
    }
    if !scalar_rhs {
        if b.shape().len() > a.shape().len() {
            return Err("cannot broadcast RHS to the input shape".into());
        }
        while b.shape().len() < a.shape().len() {
            b.metadata.shape.insert(0, 1);
            b.metadata.strides.insert(0, 0);
        }
        for axis in 0..a.shape().len() {
            if b.shape()[axis] == 1 {
                b.metadata.strides[axis] = 0;
            } else if b.shape()[axis] != a.shape()[axis] {
                return Err("cannot broadcast RHS to the input shape".into());
            }
        }
    }
    let size = count_elements(a.shape())?;
    let out = TensorHandle::new_contiguous(
        a.shape().to_vec(),
        gpu.client.empty(
            size.checked_mul(a.dtype.size())
                .ok_or("byte size overflow")?,
        ),
        a.dtype,
    );
    if size != 0 {
        let dim = CubeDim::new(&gpu.client, size);
        let count = cubecl::calculate_cube_count_elemwise(&gpu.client, size, dim);
        // All tensors originate from validated uploads/layouts; each output coordinate
        // addresses matching input axes, with zero strides only for broadcast axes.
        unsafe {
            evaluate::launch_unchecked(
                &gpu.client,
                count,
                dim,
                out.required_address_type(),
                a.clone().into_arg(),
                b.into_arg(),
                out.clone().into_arg(),
                InputScalar::new(scalar, a.dtype),
                scalar_rhs,
                op as u8,
                a.dtype,
            );
        }
    }
    Ok(out)
}

pub(crate) fn create<T: CubeElement>(
    gpu: &Gpu,
    shape: &[usize],
    identity: bool,
) -> Result<TensorHandle> {
    check_precision::<T>(gpu)?;
    if shape.is_empty() {
        return Err("at least one dimension is required".into());
    }
    count_elements(shape)?
        .checked_mul(T::cube_type().size())
        .ok_or("byte size overflow")?;
    if identity && (shape.len() != 2 || shape[0] != shape[1]) {
        return Err("identity requires a square shape".into());
    }
    let tensor = if shape.contains(&0) {
        TensorHandle::new_contiguous(shape.to_vec(), gpu.client.empty(0), T::cube_type())
    } else if identity {
        let tensor = TensorHandle::empty(&gpu.client, shape.to_vec(), T::cube_type());
        cubecl::std::tensor::identity::launch(&gpu.client, &tensor);
        tensor
    } else {
        TensorHandle::zeros(&gpu.client, shape.to_vec(), T::cube_type())
    };
    Ok(tensor)
}
