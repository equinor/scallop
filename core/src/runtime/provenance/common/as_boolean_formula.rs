use super::{count, BooleanFormula, DefaultWmcBackend, Disjunctions, IndependentFacts, Literal, Semiring, WmcBackend};

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
    count(&self.or_exactly_one_violation(disj), &IndependentFacts(v), s, backend)
  }

  /// The legacy event: the query succeeds or some group violates exactly-one.
  ///
  /// This event does not make the groups categorical choices or condition on
  /// their validity.
  fn or_exactly_one_violation(&self, disj: &Disjunctions) -> BooleanFormula {
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
    BooleanFormula::disjunction(std::iter::once(self.as_boolean_formula()).chain(violations))
  }
}
