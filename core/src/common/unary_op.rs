use serde::*;

use super::value_type::ValueType;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum UnaryOp {
  Neg,
  Pos,
  Not,
  TypeCast(ValueType),
}

impl UnaryOp {
  pub fn is_pos_neg(&self) -> bool {
    matches!(self, Self::Pos | Self::Neg)
  }

  pub fn is_not(&self) -> bool {
    matches!(self, Self::Not)
  }

  pub fn cast_to_type(&self) -> Option<&ValueType> {
    match self {
      Self::TypeCast(t) => Some(t),
      _ => None,
    }
  }
}
