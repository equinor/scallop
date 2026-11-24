use std::cmp::Ordering;

use crate::utils::FloatKey;

use super::*;

/// A symbolic value of the tensor
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorValue {
  pub shape: TensorShape,
  pub expr: TensorExpr,
}

impl TensorValue {
  pub fn is_scalar(&self) -> bool {
    self.shape.dim() == 0
  }

  /// Adds tensors of the same shape.
  pub fn checked_add(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape == v2.shape {
      Some(TensorValue {
        shape: self.shape,
        expr: self.expr + v2.expr,
      })
    } else {
      None
    }
  }

  /// Subtracts tensors of the same shape.
  pub fn checked_sub(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape == v2.shape {
      Some(TensorValue {
        shape: self.shape,
        expr: self.expr - v2.expr,
      })
    } else {
      None
    }
  }

  /// Multiplies tensors of the same shape, or broadcasts a scalar operand.
  pub fn checked_mul(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape == v2.shape || v2.is_scalar() {
      Some(TensorValue {
        shape: self.shape,
        expr: self.expr * v2.expr,
      })
    } else if self.is_scalar() {
      Some(TensorValue {
        shape: v2.shape,
        expr: self.expr * v2.expr,
      })
    } else {
      None
    }
  }

  pub fn dot(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape.dim() == 1 && self.shape == v2.shape {
      Some(TensorValue {
        shape: TensorShape::scalar(),
        expr: self.expr.dot(v2.expr),
      })
    } else {
      None
    }
  }
}

impl From<f64> for TensorValue {
  fn from(value: f64) -> Self {
    Self {
      shape: TensorShape::scalar(),
      expr: TensorExpr::Float(value),
    }
  }
}

impl From<InternalTensorSymbol> for TensorValue {
  fn from(value: InternalTensorSymbol) -> Self {
    Self {
      shape: value.shape.clone(),
      expr: TensorExpr::Symbol(value),
    }
  }
}

impl std::fmt::Display for TensorValue {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.expr.fmt(f)
  }
}

#[derive(Debug, Clone)]
pub enum TensorExpr {
  /// A base tensor
  Symbol(InternalTensorSymbol),

  /// Constant floating point
  Float(f64),

  /// A sum of two tensors
  Add(Box<TensorExpr>, Box<TensorExpr>),

  /// A subtraction between two tensors
  Sub(Box<TensorExpr>, Box<TensorExpr>),

  /// An element-wise multiplication of two tensors
  Mul(Box<TensorExpr>, Box<TensorExpr>),

  /// A dot product between two tensors
  Dot(Box<TensorExpr>, Box<TensorExpr>),
}

impl TensorExpr {
  fn variant_order(&self) -> u8 {
    match self {
      Self::Symbol(_) => 0,
      Self::Float(_) => 1,
      Self::Add(..) => 2,
      Self::Sub(..) => 3,
      Self::Mul(..) => 4,
      Self::Dot(..) => 5,
    }
  }

  pub fn dot(self, other: Self) -> Self {
    Self::Dot(Box::new(self), Box::new(other))
  }
}

impl PartialEq for TensorExpr {
  fn eq(&self, other: &Self) -> bool {
    self.cmp(other).is_eq()
  }
}

impl Eq for TensorExpr {}

impl PartialOrd for TensorExpr {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for TensorExpr {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self, other) {
      (Self::Symbol(a), Self::Symbol(b)) => a.cmp(b),
      (Self::Float(a), Self::Float(b)) => FloatKey(*a).cmp(&FloatKey(*b)),
      (Self::Add(a, b), Self::Add(c, d))
      | (Self::Sub(a, b), Self::Sub(c, d))
      | (Self::Mul(a, b), Self::Mul(c, d))
      | (Self::Dot(a, b), Self::Dot(c, d)) => (a, b).cmp(&(c, d)),
      _ => self.variant_order().cmp(&other.variant_order()),
    }
  }
}

impl std::fmt::Display for TensorExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Symbol(s) => s.fmt(f),
      Self::Float(n) => n.fmt(f),
      Self::Add(v1, v2) => f.write_fmt(format_args!("{} + {}", v1, v2)),
      Self::Sub(v1, v2) => f.write_fmt(format_args!("{} - {}", v1, v2)),
      Self::Mul(v1, v2) => f.write_fmt(format_args!("{} * {}", v1, v2)),
      Self::Dot(v1, v2) => f.write_fmt(format_args!("dot({}, {})", v1, v2)),
    }
  }
}

impl std::hash::Hash for TensorExpr {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    match self {
      Self::Symbol(s) => s.hash(state),
      Self::Float(f) => FloatKey(*f).hash(state),
      Self::Add(v1, v2) => {
        "add".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
      Self::Sub(v1, v2) => {
        "sub".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
      Self::Mul(v1, v2) => {
        "mul".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
      Self::Dot(v1, v2) => {
        "dot".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
    }
  }
}

impl From<InternalTensorSymbol> for TensorExpr {
  fn from(sym: InternalTensorSymbol) -> Self {
    Self::Symbol(sym)
  }
}

impl std::ops::Add<TensorExpr> for TensorExpr {
  type Output = TensorExpr;

  fn add(self, rhs: TensorExpr) -> Self::Output {
    TensorExpr::Add(Box::new(self), Box::new(rhs))
  }
}

impl std::ops::Sub<TensorExpr> for TensorExpr {
  type Output = TensorExpr;

  fn sub(self, rhs: TensorExpr) -> Self::Output {
    TensorExpr::Sub(Box::new(self), Box::new(rhs))
  }
}

impl std::ops::Mul<TensorExpr> for TensorExpr {
  type Output = TensorExpr;

  fn mul(self, rhs: TensorExpr) -> Self::Output {
    TensorExpr::Mul(Box::new(self), Box::new(rhs))
  }
}
