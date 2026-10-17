use crate::resident::error;
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, NdFloat};
use nu_protocol::{LabeledError, Span, Value};

fn number<T: NdFloat>(value: &Value) -> Result<T, LabeledError> {
    T::from(value.coerce_float()?)
        .ok_or_else(|| error("number cannot be represented", value.span()))
}

fn flatten<T: NdFloat>(value: &Value, data: &mut Vec<T>) -> Result<Vec<usize>, LabeledError> {
    if let Value::List { vals, .. } = value {
        let mut child_shape = None;
        for child in vals {
            let shape = flatten(child, data)?;
            if let Some(expected) = &child_shape {
                if expected != &shape {
                    return Err(error("ragged lists are not tensors", child.span()));
                }
            } else {
                child_shape = Some(shape);
            }
        }
        let mut shape = vec![vals.len()];
        shape.extend(child_shape.unwrap_or_default());
        Ok(shape)
    } else {
        data.push(number(value)?);
        Ok(vec![])
    }
}

pub(crate) fn from_nu<T: NdFloat>(
    input: &Value,
    shape: Option<Vec<usize>>,
) -> Result<ArrayD<T>, LabeledError> {
    let mut data = Vec::with_capacity(input.as_list().map_or(1, <[Value]>::len));
    let inferred = if let Value::List { vals, .. } = input
        && let Some(Value::Record { val: first, .. }) = vals.first()
    {
        for row in vals {
            let record = row.as_record()?;
            if record.len() != first.len() {
                return Err(error("table columns must match", row.span()));
            }
            for (name, _) in first.as_ref() {
                let cell = record
                    .get(name)
                    .ok_or_else(|| error("table columns must match", row.span()))?;
                data.push(number(cell)?);
            }
        }
        vec![vals.len(), first.len()]
    } else {
        flatten(input, &mut data)?
    };
    ArrayD::from_shape_vec(IxDyn(&shape.unwrap_or(inferred)), data)
        .map_err(|e| error(e.to_string(), input.span()))
}

fn list(
    values: impl ExactSizeIterator<Item = Result<Value, LabeledError>>,
    span: Span,
) -> Result<Value, LabeledError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|e| error(e.to_string(), span))?;
    for value in values {
        result.push(value?);
    }
    Ok(Value::list(result, span))
}

fn nested<T: NdFloat + Into<f64>>(view: ArrayViewD<T>, span: Span) -> Result<Value, LabeledError> {
    if view.ndim() == 0 {
        Ok(Value::float(view[IxDyn(&[])].into(), span))
    } else {
        list(view.axis_iter(Axis(0)).map(|row| nested(row, span)), span)
    }
}

pub(crate) fn into_nu<T: NdFloat + Into<f64>>(
    view: ArrayViewD<T>,
    records: bool,
    span: Span,
) -> Result<Value, LabeledError> {
    if !records {
        return nested(view, span);
    }
    if view.ndim() != 2 {
        return Err(error("--as-records requires a 2D tensor", span));
    }
    list(
        view.axis_iter(Axis(0)).map(|row| {
            Ok(Value::record(
                row.iter()
                    .enumerate()
                    .map(|(column, &value)| {
                        (format!("col{column}"), Value::float(value.into(), span))
                    })
                    .collect(),
                span,
            ))
        }),
        span,
    )
}
