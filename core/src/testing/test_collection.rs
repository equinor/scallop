use crate::common::tuple::Tuple;
use crate::common::value::Value;
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

pub struct TestCollection {
  pub elements: Vec<Tuple>,
}

impl TestCollection {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }
}

impl<T: Into<Tuple>> From<Vec<T>> for TestCollection {
  fn from(v: Vec<T>) -> Self {
    Self {
      elements: v.into_iter().map(|t| t.into()).collect(),
    }
  }
}

pub struct TestCollectionWithTag<Prov: Provenance> {
  pub elements: Vec<(OutputTagOf<Prov>, Tuple)>,
}

impl<Prov: Provenance> TestCollectionWithTag<Prov> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }
}

impl<Prov: Provenance, Tup: Into<Tuple>> From<Vec<(OutputTagOf<Prov>, Tup)>> for TestCollectionWithTag<Prov> {
  fn from(v: Vec<(OutputTagOf<Prov>, Tup)>) -> Self {
    Self {
      elements: v.into_iter().map(|(tag, tup)| (tag, tup.into())).collect(),
    }
  }
}

pub fn test_equals(t1: &Tuple, t2: &Tuple) -> bool {
  match (t1, t2) {
    (Tuple::Tuple(ts1), Tuple::Tuple(ts2)) => ts1.iter().zip(ts2.iter()).all(|(s1, s2)| test_equals(s1, s2)),
    (Tuple::Value(Value::F32(t1)), Tuple::Value(Value::F32(t2))) => t1 == t2 || (t1 - t2).abs() < 0.001,
    (Tuple::Value(Value::F64(t1)), Tuple::Value(Value::F64(t2))) => t1 == t2 || (t1 - t2).abs() < 0.001,
    _ => t1 == t2,
  }
}

pub fn expect_collection<Prov, C>(actual: &DynamicCollection<Prov>, expected: C)
where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
  C: Into<TestCollection>,
{
  let expected = Into::<TestCollection>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let te = e.clone();
    let pos = actual.iter().position(|elem| test_equals(&elem.tuple, &te));
    assert!(pos.is_some(), "Tuple {:?} not found in collection {:?}", te, actual)
  }

  // Then check everything in actual is in expected
  for elem in actual.iter() {
    let pos = expected
      .elements
      .iter()
      .position(|e| test_equals(&e.clone(), &elem.tuple));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection but not found in expected set",
      elem
    )
  }
}

pub fn expect_output_collection<Prov, C>(name: &str, actual: &DynamicOutputCollection<Prov>, expected: C)
where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
  C: Into<TestCollection>,
{
  let expected = Into::<TestCollection>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let te = e.clone();
    let pos = actual.iter().position(|(_, tuple)| test_equals(tuple, &te));
    assert!(
      pos.is_some(),
      "Tuple {:?} not found in `{}` collection {:?}",
      te,
      name,
      actual
    )
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected.elements.iter().position(|e| test_equals(&e.clone(), &elem.1));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection `{}` but not found in expected set",
      elem,
      name,
    )
  }
}

pub fn expect_contains_output_collection<Prov, C>(name: &str, actual: &DynamicOutputCollection<Prov>, expected: C)
where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
  C: Into<TestCollection>,
{
  let expected = Into::<TestCollection>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let te = e.clone();
    let pos = actual.iter().position(|(_, tuple)| test_equals(tuple, &te));
    assert!(
      pos.is_some(),
      "Tuple {:?} not found in `{}` collection {:?}",
      te,
      name,
      actual
    )
  }
}

pub fn expect_output_collection_with_tag<Prov, C, F>(
  name: &str,
  actual: &DynamicOutputCollection<Prov>,
  expected: C,
  cmp: F,
) where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
  C: Into<TestCollectionWithTag<Prov>>,
  F: Fn(&OutputTagOf<Prov>, &OutputTagOf<Prov>) -> bool,
{
  let expected = Into::<TestCollectionWithTag<Prov>>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let (tage, te) = e.clone();
    let pos = actual
      .iter()
      .position(|(tag, tuple)| test_equals(tuple, &te) && cmp(&tage, tag));
    assert!(
      pos.is_some(),
      "Tagged Tuple {:?} not found in `{}` collection {:?}",
      (tage, te),
      name,
      actual
    )
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .elements
      .iter()
      .position(|(tag, tup)| test_equals(&tup.clone(), &elem.1) && cmp(tag, &elem.0));
    assert!(
      pos.is_some(),
      "Tagged Tuple {:?} is derived in `{}` collection but not found in expected set",
      elem,
      name,
    )
  }
}
