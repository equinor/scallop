use std::cmp::Ordering;

use crate::common::tuple::*;
use crate::common::value::*;
use crate::utils::FloatKey;

pub trait StaticTupleTrait: 'static + Sized + Clone + std::fmt::Debug + std::cmp::PartialOrd {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self;

  fn into_dyn_tuple(self) -> Tuple;

  /// Compare tuple keys with the same total ordering used to sort static collections.
  ///
  /// The default requires a total `PartialOrd`; types with unordered values must override this.
  fn cmp_tuple(&self, other: &Self) -> Ordering {
    self
      .partial_cmp(other)
      .expect("static tuple keys with unordered values must implement cmp_tuple")
  }
}

impl StaticTupleTrait for () {
  fn from_dyn_tuple(_: Tuple) -> Self {}
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Tuple(Box::new([]))
  }
}

impl StaticTupleTrait for i8 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_i8()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::I8(self))
  }
}

impl StaticTupleTrait for i16 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_i16()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::I16(self))
  }
}

impl StaticTupleTrait for i32 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_i32()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::I32(self))
  }
}

impl StaticTupleTrait for i64 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_i64()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::I64(self))
  }
}

impl StaticTupleTrait for i128 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_i128()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::I128(self))
  }
}

impl StaticTupleTrait for isize {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_isize()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::ISize(self))
  }
}

impl StaticTupleTrait for u8 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_u8()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::U8(self))
  }
}

impl StaticTupleTrait for u16 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_u16()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::U16(self))
  }
}

impl StaticTupleTrait for u32 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_u32()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::U32(self))
  }
}

impl StaticTupleTrait for u64 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_u64()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::U64(self))
  }
}

impl StaticTupleTrait for u128 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_u128()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::U128(self))
  }
}

impl StaticTupleTrait for usize {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_usize()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::USize(self))
  }
}

impl StaticTupleTrait for f32 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_f32()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::F32(self))
  }
  fn cmp_tuple(&self, other: &Self) -> Ordering {
    FloatKey(*self).cmp(&FloatKey(*other))
  }
}

impl StaticTupleTrait for f64 {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_f64()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::F64(self))
  }
  fn cmp_tuple(&self, other: &Self) -> Ordering {
    FloatKey(*self).cmp(&FloatKey(*other))
  }
}

impl StaticTupleTrait for bool {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_bool()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::Bool(self))
  }
}

impl StaticTupleTrait for char {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_char()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::Char(self))
  }
}

impl StaticTupleTrait for &'static str {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_str()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::Str(self))
  }
}

impl StaticTupleTrait for String {
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    dyn_tuple.as_string()
  }
  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Value(Value::String(self))
  }
}

impl<T1> StaticTupleTrait for (T1,)
where
  T1: StaticTupleTrait,
{
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    match dyn_tuple {
      Tuple::Tuple(elems) => (T1::from_dyn_tuple(elems[0].clone()),),
      _ => panic!("expected dyn tuple"),
    }
  }

  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Tuple(Box::new([self.0.into_dyn_tuple()]))
  }

  fn cmp_tuple(&self, other: &Self) -> Ordering {
    self.0.cmp_tuple(&other.0)
  }
}

impl<T1, T2> StaticTupleTrait for (T1, T2)
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
{
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    match dyn_tuple {
      Tuple::Tuple(elems) => (
        T1::from_dyn_tuple(elems[0].clone()),
        T2::from_dyn_tuple(elems[1].clone()),
      ),
      _ => panic!("expected dyn tuple"),
    }
  }

  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Tuple(Box::new([self.0.into_dyn_tuple(), self.1.into_dyn_tuple()]))
  }

  fn cmp_tuple(&self, other: &Self) -> Ordering {
    self.0.cmp_tuple(&other.0).then_with(|| self.1.cmp_tuple(&other.1))
  }
}

impl<T1, T2, T3> StaticTupleTrait for (T1, T2, T3)
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T3: StaticTupleTrait,
{
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    match dyn_tuple {
      Tuple::Tuple(elems) => (
        T1::from_dyn_tuple(elems[0].clone()),
        T2::from_dyn_tuple(elems[1].clone()),
        T3::from_dyn_tuple(elems[2].clone()),
      ),
      _ => panic!("expected dyn tuple"),
    }
  }

  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Tuple(Box::new([
      self.0.into_dyn_tuple(),
      self.1.into_dyn_tuple(),
      self.2.into_dyn_tuple(),
    ]))
  }

  fn cmp_tuple(&self, other: &Self) -> Ordering {
    self
      .0
      .cmp_tuple(&other.0)
      .then_with(|| self.1.cmp_tuple(&other.1))
      .then_with(|| self.2.cmp_tuple(&other.2))
  }
}

impl<T1, T2, T3, T4> StaticTupleTrait for (T1, T2, T3, T4)
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T3: StaticTupleTrait,
  T4: StaticTupleTrait,
{
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    match dyn_tuple {
      Tuple::Tuple(elems) => (
        T1::from_dyn_tuple(elems[0].clone()),
        T2::from_dyn_tuple(elems[1].clone()),
        T3::from_dyn_tuple(elems[2].clone()),
        T4::from_dyn_tuple(elems[3].clone()),
      ),
      _ => panic!("expected dyn tuple"),
    }
  }

  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Tuple(Box::new([
      self.0.into_dyn_tuple(),
      self.1.into_dyn_tuple(),
      self.2.into_dyn_tuple(),
      self.3.into_dyn_tuple(),
    ]))
  }

  fn cmp_tuple(&self, other: &Self) -> Ordering {
    self
      .0
      .cmp_tuple(&other.0)
      .then_with(|| self.1.cmp_tuple(&other.1))
      .then_with(|| self.2.cmp_tuple(&other.2))
      .then_with(|| self.3.cmp_tuple(&other.3))
  }
}

impl<T1, T2, T3, T4, T5> StaticTupleTrait for (T1, T2, T3, T4, T5)
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T3: StaticTupleTrait,
  T4: StaticTupleTrait,
  T5: StaticTupleTrait,
{
  fn from_dyn_tuple(dyn_tuple: Tuple) -> Self {
    match dyn_tuple {
      Tuple::Tuple(elems) => (
        T1::from_dyn_tuple(elems[0].clone()),
        T2::from_dyn_tuple(elems[1].clone()),
        T3::from_dyn_tuple(elems[2].clone()),
        T4::from_dyn_tuple(elems[3].clone()),
        T5::from_dyn_tuple(elems[4].clone()),
      ),
      _ => panic!("expected dyn tuple"),
    }
  }

  fn into_dyn_tuple(self) -> Tuple {
    Tuple::Tuple(Box::new([
      self.0.into_dyn_tuple(),
      self.1.into_dyn_tuple(),
      self.2.into_dyn_tuple(),
      self.3.into_dyn_tuple(),
      self.4.into_dyn_tuple(),
    ]))
  }

  fn cmp_tuple(&self, other: &Self) -> Ordering {
    self
      .0
      .cmp_tuple(&other.0)
      .then_with(|| self.1.cmp_tuple(&other.1))
      .then_with(|| self.2.cmp_tuple(&other.2))
      .then_with(|| self.3.cmp_tuple(&other.3))
      .then_with(|| self.4.cmp_tuple(&other.4))
  }
}

#[derive(Clone)]
pub struct StaticTupleWrapper<T: StaticTupleTrait>(T);

impl<T: StaticTupleTrait> StaticTupleWrapper<T> {
  pub fn new(t: T) -> Self {
    Self(t)
  }

  pub fn get(&self) -> &T {
    &self.0
  }

  pub fn into(self) -> T {
    self.0
  }
}

impl<T: StaticTupleTrait> PartialEq for StaticTupleWrapper<T> {
  fn eq(&self, other: &Self) -> bool {
    self.cmp(other).is_eq()
  }
}

impl<T: StaticTupleTrait> Eq for StaticTupleWrapper<T> {}

impl<T: StaticTupleTrait> PartialOrd for StaticTupleWrapper<T> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<T: StaticTupleTrait> Ord for StaticTupleWrapper<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.0.cmp_tuple(&other.0)
  }
}

impl<T> std::ops::Deref for StaticTupleWrapper<T>
where
  T: StaticTupleTrait,
{
  type Target = T;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<T: StaticTupleTrait> std::fmt::Debug for StaticTupleWrapper<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}

impl<T: StaticTupleTrait> std::fmt::Display for StaticTupleWrapper<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}
