//! Nu commands keep f32/f64 arrays on an explicitly selected Vulkan GPU.
//!
//! Zeros and identity require positive dimensions and an explicit `--device`;
//! `--dtype` defaults to f64. Transpose reverses all axes and shares storage.
//! Reshape shares row-major contiguous storage; other layouts and mismatched
//! element counts are errors, matching native Nu matrix reshape. Flatten accepts
//! empty arrays. Metadata operations never read elements back to the host.
//!
//! Add/subtract and `+ - * /` accept a scalar or a same-shape resident RHS.
//! Scale accepts a scalar. Dtype and device must match. Only add/subtract with
//! explicit `--broadcast` expand the RHS to the LHS shape. `*` is elementwise;
//! `multiply` remains the 1D/2D dot product. Outputs never mutate their inputs.
//! Arithmetic runs in the stored dtype. Division rounding may differ from CPU
//! results; NaN payloads are not specified. No host arithmetic fallback is used.
use crate::{
    convert::{from_nu, into_nu},
    elementwise::{Arithmetic, Operand},
    gpu::{Gpu, download_array, open_device, upload_array},
    resident::{Resources, error},
};
use cubecl::{
    CubeElement,
    wgpu::{WgpuDeviceKind, wgpu},
};
use ndarray::ArrayD;
use nu_plugin::{EngineInterface, EvaluatedCall, Plugin, PluginCommand, SimplePluginCommand};
use nu_protocol::{
    CustomValue, LabeledError, Signature, Span, Spanned, SyntaxShape, Type, Value,
    ast::{Math, Operator},
    record,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

pub struct MatrixPlugin {
    resources: Mutex<Resources>,
    devices: Mutex<BTreeMap<String, Arc<Gpu>>>,
}

#[cfg(target_os = "linux")]
mod file_commands;

impl MatrixPlugin {
    pub fn new() -> Result<Self, LabeledError> {
        Ok(Self {
            resources: Mutex::new(Resources::new()?),
            devices: Mutex::new(BTreeMap::new()),
        })
    }
    fn device(&self, name: &str, span: Span) -> Result<Arc<Gpu>, LabeledError> {
        let mut devices = self
            .devices
            .lock()
            .map_err(|e| error(e.to_string(), span))?;
        if let Some(gpu) = devices.get(name) {
            return Ok(gpu.clone());
        }
        let index = name
            .get(1..)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| error("device must be dN or iN", span))?;
        let kind = match name.chars().next() {
            Some('d') if name == format!("d{index}") => WgpuDeviceKind::DiscreteGpu(index),
            Some('i') if name == format!("i{index}") => WgpuDeviceKind::IntegratedGpu(index),
            _ => return Err(error("device must be dN or iN", span)),
        };
        let gpu = Arc::new(open_device(kind).map_err(|e| error(e.to_string(), span))?);
        devices.insert(name.into(), gpu.clone());
        Ok(gpu)
    }
    fn upload<T: CubeElement>(
        &self,
        engine: &EngineInterface,
        gpu: Arc<Gpu>,
        array: ArrayD<T>,
        device: &str,
        dtype: &str,
        span: Span,
    ) -> Result<Value, LabeledError> {
        let tensor = upload_array(&gpu, array).map_err(|e| error(e.to_string(), span))?;
        self.register(engine, gpu, tensor, device, dtype, span)
    }
    fn register(
        &self,
        engine: &EngineInterface,
        gpu: Arc<Gpu>,
        tensor: cubecl::std::tensor::TensorHandle,
        device: &str,
        dtype: &str,
        span: Span,
    ) -> Result<Value, LabeledError> {
        cubecl::future::block_on(gpu.client.sync_buffers([&tensor.handle]))
            .map_err(|e| error(e.to_string(), span))?;
        let mut resources = self
            .resources
            .lock()
            .map_err(|e| error(e.to_string(), span))?;
        let first = resources.is_empty();
        let value = resources.insert(gpu, tensor, device, dtype, span)?;
        if first && let Err(failure) = engine.set_gc_disabled(true) {
            resources.remove(value.as_custom_value()?);
            return Err(failure.into());
        }
        Ok(value)
    }

    fn apply(
        &self,
        engine: &EngineInterface,
        input: &Value,
        right: &Value,
        op: Arithmetic,
        call: &EvaluatedCall,
    ) -> Result<Value, LabeledError> {
        let (a, b) = {
            let resources = self
                .resources
                .lock()
                .map_err(|e| error(e.to_string(), call.head))?;
            let a = resources.get(input, call.head)?;
            let b = match right {
                Value::Int { .. } | Value::Float { .. } => None,
                _ => Some(resources.get(right, call.head)?),
            };
            (a, b)
        };
        let rhs = match &b {
            Some(b) => Operand::Tensor(&b.tensor),
            None => Operand::Scalar(right.coerce_float()?),
        };
        let tensor =
            crate::elementwise::apply(&a.gpu, &a.tensor, rhs, op, call.has_flag("broadcast")?)
                .map_err(|e| error(e.to_string(), call.head))?;
        let dtype = describe_dtype(&tensor);
        self.register(engine, a.gpu.clone(), tensor, &a.device, dtype, call.head)
    }
}

impl Plugin for MatrixPlugin {
    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }
    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            Box::new(Devices),
            Box::new(FromNu),
            Box::new(IntoNu),
            #[cfg(target_os = "linux")]
            Box::new(file_commands::FromNpy),
            #[cfg(target_os = "linux")]
            Box::new(file_commands::ToNpy),
            Box::new(Multiply),
            Box::new(ReductionCommand::Sum),
            Box::new(ReductionCommand::Max),
            Box::new(ReductionCommand::Mean),
            Box::new(LayoutCommand::Zeros),
            Box::new(LayoutCommand::Identity),
            Box::new(LayoutCommand::Transpose),
            Box::new(LayoutCommand::Reshape),
            Box::new(ElementwiseCommand {
                name: "vulkan matrix add",
                operation: Arithmetic::Add,
            }),
            Box::new(ElementwiseCommand {
                name: "vulkan matrix subtract",
                operation: Arithmetic::Subtract,
            }),
            Box::new(ElementwiseCommand {
                name: "vulkan matrix scale",
                operation: Arithmetic::Multiply,
            }),
        ]
    }
    fn custom_value_operation(
        &self,
        engine: &EngineInterface,
        left: Spanned<Box<dyn CustomValue>>,
        operator: Spanned<Operator>,
        right: Value,
    ) -> Result<Value, LabeledError> {
        let op = match operator.item {
            Operator::Math(Math::Add) => Arithmetic::Add,
            Operator::Math(Math::Subtract) => Arithmetic::Subtract,
            Operator::Math(Math::Multiply) => Arithmetic::Multiply,
            Operator::Math(Math::Divide) => Arithmetic::Divide,
            _ => {
                return Err(error(
                    "supported element operators are + - * /",
                    operator.span,
                ));
            }
        };
        self.apply(
            engine,
            &Value::custom(left.item, left.span),
            &right,
            op,
            &EvaluatedCall::new(operator.span),
        )
    }
    fn custom_value_dropped(
        &self,
        engine: &EngineInterface,
        value: Box<dyn CustomValue>,
    ) -> Result<(), LabeledError> {
        let mut resources = self
            .resources
            .lock()
            .map_err(|e| error(e.to_string(), Span::unknown()))?;
        resources.remove(value.as_ref());
        if resources.is_empty() {
            engine.set_gc_disabled(false)?;
        }
        Ok(())
    }
}

struct ElementwiseCommand {
    name: &'static str,
    operation: Arithmetic,
}

impl SimplePluginCommand for ElementwiseCommand {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "Apply an element operation to resident GPU values."
    }
    fn signature(&self) -> Signature {
        let sig = Signature::build(self.name).input_output_type(
            Type::Custom("vulkan matrix".into()),
            Type::Custom("vulkan matrix".into()),
        );
        if matches!(self.operation, Arithmetic::Multiply) {
            sig.required("scalar", SyntaxShape::Number, "Scalar factor.")
        } else {
            sig.required("other", SyntaxShape::Any, "Resident matrix or scalar.")
                .switch("broadcast", "Broadcast RHS to the input shape.", None)
        }
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        plugin.apply(engine, input, &call.req::<Value>(0)?, self.operation, call)
    }
}

enum LayoutCommand {
    Zeros,
    Identity,
    Transpose,
    Reshape,
}

impl SimplePluginCommand for LayoutCommand {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        match self {
            Self::Zeros => "vulkan matrix zeros",
            Self::Identity => "vulkan matrix identity",
            Self::Transpose => "vulkan matrix transpose",
            Self::Reshape => "vulkan matrix reshape",
        }
    }
    fn description(&self) -> &str {
        match self {
            Self::Zeros => "Create a zero-filled array on an explicit Vulkan GPU.",
            Self::Identity => "Create an identity matrix on an explicit Vulkan GPU.",
            Self::Transpose => "Reverse all axes while sharing GPU storage.",
            Self::Reshape => "Reshape row-major GPU storage without copying elements.",
        }
    }
    fn signature(&self) -> Signature {
        let input = if matches!(self, Self::Zeros | Self::Identity) {
            Type::Nothing
        } else {
            Type::Custom("vulkan matrix".into())
        };
        let sig = Signature::build(SimplePluginCommand::name(self))
            .input_output_type(input, Type::Custom("vulkan matrix".into()));
        match self {
            Self::Zeros | Self::Identity => {
                let sig = sig
                    .required("dimension", SyntaxShape::Int, "Positive axis length.")
                    .named(
                        "device",
                        SyntaxShape::String,
                        "Explicit GPU selector (dN or iN).",
                        None,
                    )
                    .named(
                        "dtype",
                        SyntaxShape::String,
                        "f64 (default) or explicit f32.",
                        None,
                    );
                if matches!(self, Self::Zeros) {
                    sig.rest(
                        "dimensions",
                        SyntaxShape::Int,
                        "Additional positive axis lengths.",
                    )
                } else {
                    sig
                }
            }
            Self::Reshape => sig
                .rest("dimensions", SyntaxShape::Int, "Positive axis lengths.")
                .switch("flatten", "Flatten to a one-dimensional array.", Some('f')),
            Self::Transpose => sig,
        }
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        let mut dims = call.rest::<i64>(0)?;
        if matches!(self, Self::Zeros | Self::Identity) {
            if dims.is_empty() || dims.iter().any(|&n| n <= 0) {
                return Err(error("dimensions must be positive integers", call.head));
            }
            if matches!(self, Self::Identity) {
                dims.push(dims[0]);
            }
            let shape: Vec<usize> = dims.iter().map(|&n| n as usize).collect();
            let device: String = call
                .get_flag("device")?
                .ok_or_else(|| error("--device is required; select dN or iN", call.head))?;
            let dtype = call
                .get_flag::<String>("dtype")?
                .unwrap_or_else(|| "f64".into());
            let gpu = plugin.device(&device, call.head)?;
            let identity = matches!(self, Self::Identity);
            let tensor = match dtype.as_str() {
                "f64" => crate::elementwise::create::<f64>(&gpu, &shape, identity),
                "f32" => crate::elementwise::create::<f32>(&gpu, &shape, identity),
                _ => return Err(error("dtype must be f64 or f32", call.head)),
            }
            .map_err(|e| error(e.to_string(), call.head))?;
            return plugin.register(engine, gpu, tensor, &device, &dtype, call.head);
        }
        let held = plugin
            .resources
            .lock()
            .map_err(|e| error(e.to_string(), call.head))?
            .get(input, call.head)?;
        let tensor = if matches!(self, Self::Transpose) {
            crate::layout::transpose(&held.tensor)
        } else {
            let shape = if call.has_flag("flatten")? {
                vec![
                    crate::layout::count_elements(held.tensor.shape())
                        .map_err(|e| error(e.to_string(), call.head))?,
                ]
            } else {
                if dims.is_empty() || dims.iter().any(|&n| n <= 0) {
                    return Err(error(
                        "at least one positive dimension is required, or use --flatten",
                        call.head,
                    ));
                }
                dims.iter().map(|&n| n as usize).collect()
            };
            crate::layout::reshape(&held.tensor, &shape)
                .map_err(|e| error(e.to_string(), call.head))?
        };
        let dtype = describe_dtype(&tensor);
        plugin.register(
            engine,
            held.gpu.clone(),
            tensor,
            &held.device,
            dtype,
            call.head,
        )
    }
}

fn describe_dtype(tensor: &cubecl::std::tensor::TensorHandle) -> &'static str {
    if tensor.dtype == f64::cube_type() {
        "f64"
    } else {
        "f32"
    }
}

enum ReductionCommand {
    Sum,
    Max,
    Mean,
}

impl SimplePluginCommand for ReductionCommand {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        match self {
            Self::Sum => "vulkan matrix sum",
            Self::Max => "vulkan matrix max",
            Self::Mean => "vulkan matrix mean",
        }
    }
    fn description(&self) -> &str {
        "Reduce resident GPU elements; whole-array results return a Nu float."
    }
    fn signature(&self) -> Signature {
        let input = Type::Custom("vulkan matrix".into());
        let types = if matches!(self, Self::Mean) {
            vec![(input, Type::Float)]
        } else {
            vec![(input.clone(), Type::Float), (input.clone(), input)]
        };
        let signature = Signature::build(SimplePluginCommand::name(self)).input_output_types(types);
        if matches!(self, Self::Mean) {
            signature
        } else {
            signature.named(
                "axis",
                SyntaxShape::Int,
                "Reduce this axis and keep the result on the GPU.",
                Some('a'),
            )
        }
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        let a = plugin
            .resources
            .lock()
            .map_err(|e| error(e.to_string(), call.head))?
            .get(input, call.head)?;
        let axis = call.get_flag::<i64>("axis")?.map(|axis| axis as usize);
        let tensor = match self {
            Self::Sum => crate::reduction::sum(&a.gpu, &a.tensor, axis),
            Self::Max => crate::reduction::max(&a.gpu, &a.tensor, axis),
            Self::Mean => crate::reduction::mean(&a.gpu, &a.tensor),
        }
        .map_err(|e| error(e.to_string(), call.head))?;
        if axis.is_none() {
            return download_value(&a.gpu, &tensor, false, call.head);
        }
        let dtype = describe_dtype(&tensor);
        plugin.register(engine, a.gpu.clone(), tensor, &a.device, dtype, call.head)
    }
}

struct Multiply;
impl SimplePluginCommand for Multiply {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        "vulkan matrix multiply"
    }
    fn description(&self) -> &str {
        "Multiply resident Vulkan matrices using dot product."
    }
    fn signature(&self) -> Signature {
        Signature::build(SimplePluginCommand::name(self))
            .input_output_type(Type::Custom("vulkan matrix".into()), Type::Any)
            .required("other", SyntaxShape::Any, "The other resident matrix.")
            .switch("swap", "Swap the first two operands.", Some('s'))
            .switch(
                "multall",
                "Multiply additional arrays from left to right.",
                Some('a'),
            )
            .rest(
                "rest",
                SyntaxShape::Any,
                "Additional resident arrays (used with --multall).",
            )
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        let other: Value = call.req(0)?;
        let (mut a, mut b) = {
            let resources = plugin
                .resources
                .lock()
                .map_err(|e| error(e.to_string(), call.head))?;
            (
                resources.get(input, call.head)?,
                resources.get(&other, call.head)?,
            )
        };
        if call.has_flag("swap")? {
            std::mem::swap(&mut a, &mut b);
        }
        let mut tensor = crate::matmul::multiply(&a.gpu, &a.tensor, &b.tensor)
            .map_err(|e| error(e.to_string(), call.head))?;
        if call.has_flag("multall")? {
            for value in call.rest::<Value>(1)? {
                let next = plugin
                    .resources
                    .lock()
                    .map_err(|e| error(e.to_string(), call.head))?
                    .get(&value, call.head)?;
                tensor = crate::matmul::multiply(&a.gpu, &tensor, &next.tensor)
                    .map_err(|e| error(e.to_string(), call.head))?;
            }
        }
        if tensor.shape().is_empty() {
            return download_value(&a.gpu, &tensor, false, call.head);
        }
        let dtype = describe_dtype(&tensor);
        plugin.register(engine, a.gpu.clone(), tensor, &a.device, dtype, call.head)
    }
}

struct Devices;
impl SimplePluginCommand for Devices {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        "vulkan matrix devices"
    }
    fn description(&self) -> &str {
        "List discrete and integrated Vulkan GPUs and their explicit selectors."
    }
    fn signature(&self) -> Signature {
        Signature::build(SimplePluginCommand::name(self))
            .input_output_type(Type::Nothing, Type::table())
    }
    fn run(
        &self,
        _plugin: &MatrixPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: &Value,
    ) -> Result<Value, LabeledError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters =
            cubecl::future::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let mut counts = [0_usize; 2];
        let mut rows = Vec::new();
        for adapter in adapters {
            let info = adapter.get_info();
            let (prefix, slot) = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => ("d", 0),
                wgpu::DeviceType::IntegratedGpu => ("i", 1),
                _ => continue,
            };
            let selector = format!("{prefix}{}", counts[slot]);
            counts[slot] += 1;
            rows.push(Value::record(
                record! {
                    "device" => Value::string(selector, call.head),
                    "name" => Value::string(info.name, call.head),
                },
                call.head,
            ));
        }
        Ok(Value::list(rows, call.head))
    }
}

struct FromNu;
impl SimplePluginCommand for FromNu {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        "vulkan matrix from-nu"
    }
    fn description(&self) -> &str {
        "Upload numbers to an explicitly selected Vulkan GPU."
    }
    fn signature(&self) -> Signature {
        Signature::build(SimplePluginCommand::name(self))
            .input_output_type(Type::Any, Type::Custom("vulkan matrix".into()))
            .named(
                "device",
                SyntaxShape::String,
                "Explicit GPU selector (dN or iN).",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "f64 (default) or explicit f32.",
                None,
            )
            .named(
                "shape",
                SyntaxShape::List(Box::new(SyntaxShape::Int)),
                "Explicit shape for reshaping or ambiguous empty lists.",
                None,
            )
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        let device: String = call
            .get_flag("device")?
            .ok_or_else(|| error("--device is required; select dN or iN", call.head))?;
        let dtype = call
            .get_flag::<String>("dtype")?
            .unwrap_or_else(|| "f64".into());
        let shape = call.get_flag::<Vec<usize>>("shape")?;
        match dtype.as_str() {
            "f64" => {
                let array = from_nu::<f64>(input, shape)?;
                plugin.upload(
                    engine,
                    plugin.device(&device, call.head)?,
                    array,
                    &device,
                    &dtype,
                    call.head,
                )
            }
            "f32" => {
                let array = from_nu::<f32>(input, shape)?;
                plugin.upload(
                    engine,
                    plugin.device(&device, call.head)?,
                    array,
                    &device,
                    &dtype,
                    call.head,
                )
            }
            _ => Err(error("dtype must be f64 or f32", call.head)),
        }
    }
}

struct IntoNu;
impl SimplePluginCommand for IntoNu {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        "vulkan matrix into-nu"
    }
    fn description(&self) -> &str {
        "Explicitly download a Vulkan matrix."
    }
    fn signature(&self) -> Signature {
        Signature::build(SimplePluginCommand::name(self))
            .input_output_type(Type::Custom("vulkan matrix".into()), Type::Any)
            .switch("as-records", "Export a 2D tensor as a colN table.", None)
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        let held = plugin
            .resources
            .lock()
            .map_err(|e| error(e.to_string(), call.head))?
            .get(input, call.head)?;
        let records = call.has_flag("as-records")?;
        download_value(&held.gpu, &held.tensor, records, call.head)
    }
}

fn download_value(
    gpu: &Gpu,
    tensor: &cubecl::std::tensor::TensorHandle,
    records: bool,
    span: Span,
) -> Result<Value, LabeledError> {
    if tensor.dtype == <f64 as CubeElement>::cube_type() {
        let array = download_array::<f64>(gpu, tensor).map_err(|e| error(e.to_string(), span))?;
        into_nu(array.view(), records, span)
    } else {
        let array = download_array::<f32>(gpu, tensor).map_err(|e| error(e.to_string(), span))?;
        into_nu(array.view(), records, span)
    }
}
