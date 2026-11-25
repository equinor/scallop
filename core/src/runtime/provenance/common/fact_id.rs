use super::{AsBooleanFormula, BooleanFormula, Literal};

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FactId(pub usize);

impl AsBooleanFormula for FactId {
  fn as_boolean_formula(&self) -> BooleanFormula {
    BooleanFormula::Literal(Literal::Pos(self.0))
  }
}
