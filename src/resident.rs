use crate::gpu::Gpu;
use cubecl::std::tensor::TensorHandle;
use nu_protocol::{
    CustomValue, LabeledError, ShellError, Span, Value, ast::PathMember, casing::Casing, record,
};
use serde::{Deserialize, Serialize};
use std::{any::Any, collections::BTreeMap, sync::Arc};

pub(crate) fn error(message: impl Into<String>, span: Span) -> LabeledError {
    LabeledError::new("Vulkan matrix error").with_label(message, span)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MatrixValue {
    generation: [u8; 16],
    id: u64,
    shape: Vec<usize>,
    dtype: String,
    device: Arc<str>,
}

#[typetag::serde(name = "vulkan-matrix")]
impl CustomValue for MatrixValue {
    fn clone_value(&self, span: Span) -> Value {
        Value::custom(Box::new(self.clone()), span)
    }
    fn type_name(&self) -> String {
        "vulkan matrix".into()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }
    fn notify_plugin_on_drop(&self) -> bool {
        true
    }
    fn to_base_value(&self, span: Span) -> Result<Value, ShellError> {
        Ok(Value::record(
            record! {
                "shape" => Value::list(self.shape.iter().map(|&n| Value::int(n as i64, span)).collect(), span),
                "dtype" => Value::string(&self.dtype, span),
                "device" => Value::string(self.device.as_ref(), span),
            },
            span,
        ))
    }
    fn follow_path_string(
        &self,
        self_span: Span,
        val: String,
        span: Span,
        optional: bool,
        casing: Casing,
    ) -> Result<Value, ShellError> {
        self.to_base_value(self_span)?
            .follow_cell_path(&[PathMember::String {
                val,
                span,
                optional,
                casing,
            }])
            .map(|v| v.into_owned())
    }
}

pub(crate) struct Resident {
    pub gpu: Arc<Gpu>,
    pub tensor: TensorHandle,
    pub device: Arc<str>,
}

pub(crate) struct Resources {
    generation: [u8; 16],
    next: u64,
    entries: BTreeMap<u64, Arc<Resident>>,
}

impl Resources {
    pub fn new() -> Result<Self, LabeledError> {
        let mut generation = [0; 16];
        getrandom::fill(&mut generation).map_err(|e| error(e.to_string(), Span::unknown()))?;
        Ok(Self {
            generation,
            next: 0,
            entries: BTreeMap::new(),
        })
    }
    pub fn insert(
        &mut self,
        gpu: Arc<Gpu>,
        tensor: TensorHandle,
        device: &str,
        dtype: &str,
        span: Span,
    ) -> Result<Value, LabeledError> {
        let id = self.next;
        self.next = id
            .checked_add(1)
            .ok_or_else(|| error("resource IDs exhausted", span))?;
        let device: Arc<str> = device.into();
        let value = MatrixValue {
            generation: self.generation,
            id,
            shape: tensor.shape().to_vec(),
            dtype: dtype.into(),
            device: device.clone(),
        };
        self.entries.insert(
            id,
            Arc::new(Resident {
                gpu,
                tensor,
                device,
            }),
        );
        Ok(Value::custom(Box::new(value), span))
    }
    pub fn get(&self, value: &Value, span: Span) -> Result<Arc<Resident>, LabeledError> {
        let value = value
            .as_custom_value()
            .ok()
            .and_then(|v| v.as_any().downcast_ref::<MatrixValue>())
            .ok_or_else(|| error("expected a Vulkan matrix", span))?;
        if value.generation != self.generation {
            return Err(error(
                "matrix belongs to an earlier plugin process; upload it again",
                span,
            ));
        }
        self.entries
            .get(&value.id)
            .cloned()
            .ok_or_else(|| error("matrix resource has been released", span))
    }
    pub fn remove(&mut self, value: &dyn CustomValue) {
        if let Some(value) = value.as_any().downcast_ref::<MatrixValue>()
            && value.generation == self.generation
        {
            self.entries.remove(&value.id);
        }
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
