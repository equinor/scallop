use serde::*;
use std::cmp::Ordering;

use crate::utils::FloatKey;

use super::foreign_tensor::DynamicExternalTensor;

#[derive(Clone, Debug, Serialize, Default)]
pub enum DynamicInputTag {
  #[default]
  None,
  NewVariable,
  Exclusive(usize),
  Bool(bool),
  Natural(usize),
  Float(f64),
  ExclusiveFloat(f64, usize),
  FloatWithID(usize, f64),
  ExclusiveFloatWithID(usize, f64, usize),
  Tensor(DynamicExternalTensor),
}

impl DynamicInputTag {
  fn variant_order(&self) -> u8 {
    match self {
      Self::None => 0,
      Self::NewVariable => 1,
      Self::Exclusive(_) => 2,
      Self::Bool(_) => 3,
      Self::Natural(_) => 4,
      Self::Float(_) => 5,
      Self::ExclusiveFloat(..) => 6,
      Self::FloatWithID(..) => 7,
      Self::ExclusiveFloatWithID(..) => 8,
      Self::Tensor(_) => 9,
    }
  }

  pub fn is_some(&self) -> bool {
    !matches!(self, Self::None)
  }

  pub fn is_none(&self) -> bool {
    matches!(self, Self::None)
  }

  pub fn with_exclusivity(&self, exclusion_id: usize) -> Self {
    match self {
      Self::None => Self::Exclusive(exclusion_id),
      Self::Float(f) => Self::ExclusiveFloat(*f, exclusion_id),
      Self::ExclusiveFloat(f, _) => Self::ExclusiveFloat(*f, exclusion_id),
      _ => self.clone(),
    }
  }
}

impl PartialEq for DynamicInputTag {
  fn eq(&self, other: &Self) -> bool {
    self.cmp(other).is_eq()
  }
}

impl Eq for DynamicInputTag {}

impl PartialOrd for DynamicInputTag {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for DynamicInputTag {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self, other) {
      (Self::Exclusive(a), Self::Exclusive(b)) | (Self::Natural(a), Self::Natural(b)) => a.cmp(b),
      (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
      (Self::Float(a), Self::Float(b)) => FloatKey(*a).cmp(&FloatKey(*b)),
      (Self::ExclusiveFloat(a, id_a), Self::ExclusiveFloat(b, id_b)) => (FloatKey(*a), id_a).cmp(&(FloatKey(*b), id_b)),
      (Self::FloatWithID(id_a, a), Self::FloatWithID(id_b, b)) => (id_a, FloatKey(*a)).cmp(&(id_b, FloatKey(*b))),
      (Self::ExclusiveFloatWithID(id_a, a, ex_a), Self::ExclusiveFloatWithID(id_b, b, ex_b)) => {
        (id_a, FloatKey(*a), ex_a).cmp(&(id_b, FloatKey(*b), ex_b))
      }
      (Self::Tensor(a), Self::Tensor(b)) => a.cmp(b),
      _ => self.variant_order().cmp(&other.variant_order()),
    }
  }
}

impl std::fmt::Display for DynamicInputTag {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::None => Ok(()),
      Self::NewVariable => f.write_str("new Var"),
      Self::Exclusive(i) => f.write_fmt(format_args!("[ME({})]", i)),
      Self::Bool(b) => b.fmt(f),
      Self::Natural(n) => n.fmt(f),
      Self::Float(n) => n.fmt(f),
      Self::ExclusiveFloat(n, i) => f.write_str(&format!("{} [ME({})]", n, i)),
      Self::FloatWithID(id, n) => f.write_str(&format!("{} [ID({})]", n, id)),
      Self::ExclusiveFloatWithID(id, n, i) => f.write_str(&format!("{} [ID({}), ME({})]", n, id, i)),
      Self::Tensor(t) => t.fmt(f),
    }
  }
}

impl std::str::FromStr for DynamicInputTag {
  type Err = ParseInputTagError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if s == "true" {
      Ok(Self::Bool(true))
    } else if s == "false" {
      Ok(Self::Bool(false))
    } else {
      let f = s.parse::<f64>().map_err(|_| ParseInputTagError {
        source_str: s.to_string(),
      })?;
      Ok(Self::Float(f))
    }
  }
}

pub struct ParseInputTagError {
  source_str: String,
}

impl std::fmt::Debug for ParseInputTagError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "[Parse Input Tag Error] Cannot parse `{}` into input tag",
      self.source_str
    ))
  }
}
