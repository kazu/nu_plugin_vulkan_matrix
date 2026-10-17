use crate::convert::{from_nu, into_nu};
use ndarray::{ArrayD, IxDyn};
use nu_protocol::{Span, Value, record};

#[test]
fn scalar_nd_and_nonstandard_arrays_round_trip() {
    let span = Span::unknown();
    let scalar = from_nu::<f64>(&Value::int(5, span), None).unwrap();
    assert!(scalar.shape().is_empty());
    assert_eq!(
        into_nu(scalar.view(), false, span).unwrap(),
        Value::float(5.0, span)
    );
    let array = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), (0..8).map(f64::from).collect()).unwrap();
    for view in [array.view(), array.view().permuted_axes(IxDyn(&[2, 0, 1]))] {
        let value = into_nu(view.clone(), false, span).unwrap();
        assert_eq!(from_nu::<f64>(&value, None).unwrap(), view.to_owned());
    }
    let empty = ArrayD::<f64>::zeros(IxDyn(&[2, 0, 3]));
    let value = into_nu(empty.view(), false, span).unwrap();
    assert_eq!(
        from_nu::<f64>(&value, Some(empty.shape().to_vec())).unwrap(),
        empty
    );
}

#[test]
fn tables_use_first_row_column_order_and_export_col_names() {
    let s = Span::unknown();
    let table = Value::list(
        vec![
            Value::record(record! {"a" => Value::int(1,s), "b" => Value::int(2,s)}, s),
            Value::record(record! {"b" => Value::int(4,s), "a" => Value::int(3,s)}, s),
        ],
        s,
    );
    let array = from_nu::<f64>(&table, None).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(
        array.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let output = into_nu(array.view(), true, s).unwrap();
    assert_eq!(
        output.as_list().unwrap()[0]
            .as_record()
            .unwrap()
            .get("col1")
            .unwrap(),
        &Value::float(2.0, s)
    );
}

#[test]
fn invalid_shapes_types_and_record_outputs_are_labeled_errors() {
    let s = Span::unknown();
    let one = Value::list(vec![Value::int(1, s)], s);
    let ragged = Value::list(vec![one.clone(), Value::list(vec![], s)], s);
    for value in [ragged, Value::bool(true, s)] {
        assert!(!from_nu::<f64>(&value, None).unwrap_err().labels.is_empty());
    }
    assert!(from_nu::<f64>(&one, Some(vec![2])).is_err());
    let bad_table = Value::list(
        vec![
            Value::record(record! {"a" => Value::int(1,s)}, s),
            Value::record(record! {"b" => Value::int(2,s)}, s),
        ],
        s,
    );
    assert!(from_nu::<f64>(&bad_table, None).is_err());
    assert!(into_nu(ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2])).view(), true, s).is_err());
}

#[test]
fn f32_conversion_keeps_subnormal_and_signed_zero_bits() {
    let s = Span::unknown();
    let input = Value::list(
        vec![
            Value::float(-0.0, s),
            Value::float(f64::from(f32::from_bits(1)), s),
        ],
        s,
    );
    let array = from_nu::<f32>(&input, None).unwrap();
    assert_eq!(array[IxDyn(&[0])].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(array[IxDyn(&[1])].to_bits(), 1);
    let output = into_nu(array.view(), false, s).unwrap();
    assert_eq!(
        output.as_list().unwrap()[1].as_float().unwrap().to_bits(),
        f64::from(f32::from_bits(1)).to_bits()
    );
}

#[test]
fn enormous_empty_shapes_cannot_overflow_nu_list_allocation() {
    let array = ArrayD::<f64>::from_shape_vec(IxDyn(&[isize::MAX as usize, 0]), vec![]).unwrap();
    assert!(into_nu(array.view(), false, Span::unknown()).is_err());
    assert!(into_nu(array.view(), true, Span::unknown()).is_err());
}
