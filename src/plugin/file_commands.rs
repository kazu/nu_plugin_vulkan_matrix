use super::*;
use std::path::PathBuf;

fn path(engine: &EngineInterface, call: &EvaluatedCall) -> Result<PathBuf, LabeledError> {
    let path = PathBuf::from(call.req::<String>(0)?);
    Ok(if path.is_absolute() {
        path
    } else {
        PathBuf::from(engine.get_current_dir()?).join(path)
    })
}

pub(super) struct FromNpy;
impl SimplePluginCommand for FromNpy {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        "vulkan matrix from-npy"
    }
    fn description(&self) -> &str {
        "Import a C-order f32/f64 NumPy file without a userspace payload copy."
    }
    fn signature(&self) -> Signature {
        Signature::build(SimplePluginCommand::name(self))
            .input_output_type(Type::Nothing, Type::Custom("vulkan matrix".into()))
            .required("path", SyntaxShape::Filepath, "NumPy .npy input file.")
            .named("device", SyntaxShape::String, "Explicit GPU selector (dN or iN).", None)
            .switch("shared", "Keep a read-only file mapping as the GPU tensor; requires an aligned header. Keep the file unchanged while in use.", None)
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: &Value,
    ) -> Result<Value, LabeledError> {
        let device: String = call
            .get_flag("device")?
            .ok_or_else(|| error("--device is required; select dN or iN", call.head))?;
        let gpu = plugin.device(&device, call.head)?;
        let tensor = crate::file_io::load(&gpu, &path(engine, call)?, call.has_flag("shared")?)
            .map_err(|e| error(e.to_string(), call.head))?;
        let dtype = if tensor.dtype == <f64 as cubecl::CubeElement>::cube_type() {
            "f64"
        } else {
            "f32"
        };
        plugin.register(engine, gpu, tensor, &device, dtype, call.head)
    }
}

pub(super) struct ToNpy;
impl SimplePluginCommand for ToNpy {
    type Plugin = MatrixPlugin;
    fn name(&self) -> &str {
        "vulkan matrix to-npy"
    }
    fn description(&self) -> &str {
        "Write GPU matrix data directly into NumPy file pages and sync the file."
    }
    fn signature(&self) -> Signature {
        Signature::build(SimplePluginCommand::name(self))
            .input_output_type(Type::Custom("vulkan matrix".into()), Type::Nothing)
            .required("path", SyntaxShape::Filepath, "NumPy .npy output file.")
            .switch("force", "Atomically replace an existing output file.", None)
    }
    fn run(
        &self,
        plugin: &MatrixPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        let resident = plugin
            .resources
            .lock()
            .map_err(|e| error(e.to_string(), call.head))?
            .get(input, call.head)?;
        crate::file_io::save(
            &resident.gpu,
            &resident.tensor,
            &path(engine, call)?,
            call.has_flag("force")?,
        )
        .map_err(|e| error(e.to_string(), call.head))?;
        Ok(Value::nothing(call.head))
    }
}
