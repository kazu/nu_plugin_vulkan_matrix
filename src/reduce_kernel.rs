use cubecl::{
    prelude::*,
    std::tensor::r#virtual::{
        VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand,
    },
};
use cubek_reduce::{
    components::{
        args::{TensorArgs, init_tensors},
        instructions::ReduceOperationConfig,
    },
    launch::reduce_kernel_virtual,
    routines::ReduceBlueprint,
};

#[derive(CubeType, Clone)]
struct IgnoreNan<F: Float, N: Size> {
    input: VirtualTensor<F, N>,
}

impl<F: Float, N: Size> Vectorized for IgnoreNan<F, N> {}
impl<F: Float, N: Size> VectorizedExpand for IgnoreNanExpand<F, N> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.input.__expand_vector_size_method(scope)
    }
}

#[cube]
impl<F: Float, N: Size> VirtualTensorOperations<F, N> for IgnoreNan<F, N> {
    fn as_tensor_map(&self) -> ComptimeOption<TensorMap<F, Tiled>> {
        ComptimeOption::new_None()
    }
    fn read_window(&self, _start: usize, _end: usize) -> &[Vector<F, N>] {
        panic!("reduction reads individual vectors")
    }
    fn write(&mut self, _index: usize, _value: Vector<F, N>) {
        panic!("reduction input is read-only")
    }
    fn read(&self, index: usize) -> Vector<F, N> {
        let value = self.input.read(index);
        let identity = Vector::empty().fill(F::cast_from(f32::reinterpret(0xff80_0000u32)));
        select_many(value.equal(&value), value, identity)
    }
    fn shape(&self, axis: usize) -> usize {
        self.input.shape(axis)
    }
    fn stride(&self, axis: usize) -> usize {
        self.input.stride(axis)
    }
    fn rank(&self) -> usize {
        self.input.rank()
    }
    fn len(&self) -> usize {
        self.input.len()
    }
    fn buffer_len(&self) -> usize {
        self.input.buffer_len()
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn evaluate<F: Float, N: Size>(
    input: &OwnedTensor<Vector<F, N>>,
    output: &mut OwnedTensor<Vector<F, N>>,
    axis: usize,
    out_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] operation: ReduceOperationConfig,
    #[define(F)] _dtype: ElemType,
) {
    let (input, mut output) = init_tensors::<TensorArgs, F, N, F, N>(input, output);
    let input = if comptime!(matches!(operation, ReduceOperationConfig::Max)) {
        VirtualTensor::<F, N>::new::<IgnoreNan<F, N>>(IgnoreNan::<F, N> { input })
    } else {
        input
    };
    reduce_kernel_virtual::<F, N, F, N, F>(
        &input,
        &mut output,
        axis,
        out_axis,
        blueprint,
        operation,
    );
}

pub(crate) fn launch(
    gpu: &crate::gpu::Gpu,
    input: &cubecl::std::tensor::TensorHandle,
    output: &cubecl::std::tensor::TensorHandle,
    axis: usize,
    operation: ReduceOperationConfig,
) -> crate::gpu::Result<()> {
    use cubek_reduce::{
        ReduceDtypes, VectorizationMode, output_vectorization_axis,
        routines::{
            BlueprintStrategy, ReduceProblem, ReduceVectorSettings, Routine,
            cube::{CubeRoutine, CubeStrategy},
        },
    };
    let mode = if input.strides()[axis] == 1 {
        VectorizationMode::Parallel
    } else {
        VectorizationMode::Perpendicular
    };
    let output_axis = output_vectorization_axis(input.strides(), axis, mode);
    let address_type = input
        .required_address_type()
        .max(output.required_address_type());
    let problem = ReduceProblem {
        reduce_len: input.shape()[axis],
        reduce_count: crate::layout::count_elements(output.shape())?,
        axis,
        dtypes: ReduceDtypes {
            input: input.dtype,
            output: input.dtype,
            accumulation: input.dtype,
        },
        instruction: operation,
        address_type,
    };
    let vectors = ReduceVectorSettings {
        vectorization_mode: mode,
        vector_size_input: 1,
        vector_size_output: 1,
        unchecked_fast_paths: false,
        fuse_on_read: false,
    };
    let (blueprint, settings) = CubeRoutine.prepare(
        &gpu.client,
        problem,
        vectors,
        BlueprintStrategy::Inferred(CubeStrategy { use_planes: false }),
    )?;
    // The caller provides a nonempty valid tensor and an output with this axis set to one.
    unsafe {
        evaluate::launch_unchecked(
            &gpu.client,
            settings.cube_count,
            settings.cube_dim,
            address_type,
            1,
            input.clone().into_arg(),
            output.clone().into_arg(),
            axis,
            output_axis,
            blueprint,
            operation,
            input.dtype,
        );
    }
    Ok(())
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn fill<F: Float>(out: &mut Tensor<F>, value: InputScalar, #[define(F)] _dtype: ElemType) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = value.get::<F>();
    }
}

pub(crate) fn launch_fill(
    gpu: &crate::gpu::Gpu,
    output: &cubecl::std::tensor::TensorHandle,
    value: f64,
) {
    let size = output.shape().iter().product();
    let dim = CubeDim::new(&gpu.client, size);
    let count = cubecl::calculate_cube_count_elemwise(&gpu.client, size, dim);
    // Each unit writes one element of the fresh contiguous output.
    unsafe {
        fill::launch_unchecked(
            &gpu.client,
            count,
            dim,
            output.required_address_type(),
            output.clone().into_arg(),
            InputScalar::new(value, output.dtype),
            output.dtype,
        );
    }
}
