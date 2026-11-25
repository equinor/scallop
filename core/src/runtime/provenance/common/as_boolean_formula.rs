use super::{BooleanFormula, DefaultWmcBackend, Disjunctions, Literal, Semiring, WmcBackend};

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
    backend.wmc(&self.as_boolean_formula(), s, v)
  }

  /// Count assignments satisfying the formula or violating any exactly-one constraint.
  fn wmc_with_disjunctions<S, V>(&self, s: &S, v: &V, disj: &Disjunctions) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    self.wmc_with_disjunctions_using(&DefaultWmcBackend, s, v, disj)
  }

  /// Count the query or any exactly-one violation using a caller-supplied backend.
  fn wmc_with_disjunctions_using<S, V>(
    &self,
    backend: &impl WmcBackend,
    s: &S,
    v: &V,
    disj: &Disjunctions,
  ) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    let violations = disj.iter().map(|group| {
      BooleanFormula::conjunction(group.facts.iter().map(|negated| {
        BooleanFormula::disjunction(group.facts.iter().map(|fact| {
          let literal = if fact == negated {
            Literal::Neg(*fact)
          } else {
            Literal::Pos(*fact)
          };
          BooleanFormula::Literal(literal)
        }))
      }))
    });
    let formula = BooleanFormula::disjunction(std::iter::once(self.as_boolean_formula()).chain(violations));
    backend.wmc(&formula, s, v)
  }
}
