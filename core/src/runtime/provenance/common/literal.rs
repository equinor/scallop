use super::{AsBooleanFormula, BooleanFormula};

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum Literal {
  Pos(usize),
  Neg(usize),
}

impl Literal {
  pub fn fact_id(&self) -> usize {
    match self {
      Self::Pos(f) => *f,
      Self::Neg(f) => *f,
    }
  }

  pub fn sign(&self) -> bool {
    match self {
      Self::Pos(_) => true,
      Self::Neg(_) => false,
    }
  }

  pub fn negate(&self) -> Self {
    match self {
      Self::Pos(v) => Self::Neg(*v),
      Self::Neg(v) => Self::Pos(*v),
    }
  }
}

impl AsBooleanFormula for Literal {
  fn as_boolean_formula(&self) -> BooleanFormula {
    BooleanFormula::Literal(self.clone())
  }
}

impl std::fmt::Display for Literal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Pos(i) => f.write_fmt(format_args!("pos({})", i)),
      Self::Neg(i) => f.write_fmt(format_args!("neg({})", i)),
    }
  }
}
