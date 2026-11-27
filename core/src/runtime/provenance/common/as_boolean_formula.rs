use super::{count, BooleanFormula, DefaultWmcBackend, IndependentFacts, Semiring, WmcBackend};

pub trait AsBooleanFormula {
  fn as_boolean_formula(&self) -> BooleanFormula;

  /// Compute the weighted model count of the formula.
  fn wmc<S, V>(&self, s: &S, v: &V) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    self.wmc_using(&DefaultWmcBackend, s, v)
  }

  /// Compute the weighted model count using a caller-supplied backend.
  fn wmc_using<S, V>(&self, backend: &impl WmcBackend, s: &S, v: &V) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    count(&self.as_boolean_formula(), &IndependentFacts(v), s, backend)
  }
}
