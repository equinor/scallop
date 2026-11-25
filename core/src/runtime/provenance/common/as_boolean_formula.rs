use super::{BooleanFormula, Disjunctions, Literal, Semiring};

pub trait AsBooleanFormula {
  fn as_boolean_formula(&self) -> BooleanFormula;

  /// Compute the weighted model count of the formula.
  fn wmc<S, V>(&self, s: &S, v: &V) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    self.as_boolean_formula().wmc(s, v)
  }

  /// Count assignments satisfying the formula or violating any exactly-one constraint.
  fn wmc_with_disjunctions<S, V>(&self, s: &S, v: &V, disj: &Disjunctions) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    let violations = disj.disjunctions.iter().map(|group| {
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
    BooleanFormula::disjunction(std::iter::once(self.as_boolean_formula()).chain(violations)).wmc(s, v)
  }
}
