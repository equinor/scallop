use std::collections::{BTreeSet, HashSet};

use scallop_core::common::foreign_tensor::{InternalTensorSymbol, TensorValue};
use scallop_core::common::value::*;

#[test]
fn value_try_into_1() {
  let v = Value::USize(10);
  let p: usize = v.try_into().unwrap();
  assert_eq!(p, 10);
}

#[test]
fn value_try_into_2() {
  let v = Value::I8(10);
  let result: Result<usize, _> = v.try_into();
  assert!(matches!(result, Err(ValueConversionError)));
}

#[test]
fn value_try_from_string() {
  let value = Value::String("hello".to_string());
  assert_eq!(String::try_from(value).unwrap(), "hello");
  assert!(matches!(
    String::try_from(Value::Str("hello")),
    Err(ValueConversionError)
  ));
}

#[test]
fn tensor_multiplication_accepts_a_scalar_on_either_side() {
  let vector = TensorValue::from(InternalTensorSymbol::new(vec![3].into(), 0));
  let scalar = TensorValue::from(2.0);
  let left = scalar.clone().checked_mul(vector.clone()).unwrap();
  let right = vector.clone().checked_mul(scalar.clone()).unwrap();
  assert_eq!(left.shape, vector.shape);
  assert_eq!(right.shape, vector.shape);
  assert_eq!(left.expr, scalar.expr.clone() * vector.expr.clone());
  assert_eq!(right.expr, vector.expr * scalar.expr);
}

#[test]
fn checked_tensor_arithmetic_rejects_mismatched_shapes() {
  let lhs = TensorValue::from(InternalTensorSymbol::new(vec![2].into(), 0));
  let rhs = TensorValue::from(InternalTensorSymbol::new(vec![3].into(), 1));
  assert!(lhs.clone().checked_add(rhs.clone()).is_none());
  assert!(lhs.clone().checked_sub(rhs.clone()).is_none());
  assert!(lhs.checked_mul(rhs).is_none());
}

#[test]
fn float_value_keys_keep_nan_distinct_from_numbers_and_deduplicate_signed_zero() {
  for values in [
    [
      Value::F32(f32::NAN),
      Value::F32(-f32::NAN),
      Value::F32(0.0),
      Value::F32(-0.0),
      Value::F32(f32::INFINITY),
    ],
    [
      Value::F64(f64::NAN),
      Value::F64(-f64::NAN),
      Value::F64(0.0),
      Value::F64(-0.0),
      Value::F64(f64::INFINITY),
    ],
    [
      Value::TensorValue(TensorValue::from(f64::NAN)),
      Value::TensorValue(TensorValue::from(-f64::NAN)),
      Value::TensorValue(TensorValue::from(0.0)),
      Value::TensorValue(TensorValue::from(-0.0)),
      Value::TensorValue(TensorValue::from(f64::INFINITY)),
    ],
  ] {
    let ordered = values.iter().collect::<BTreeSet<_>>();
    let hashed = values.iter().collect::<HashSet<_>>();
    assert_eq!(ordered.len(), 3);
    assert_eq!(hashed.len(), 3);
    assert_eq!(
      ordered.into_iter().collect::<Vec<_>>(),
      [&values[2], &values[4], &values[0]]
    );
  }
}
