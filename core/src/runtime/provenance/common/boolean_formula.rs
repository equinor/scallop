use super::Literal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BooleanFormula {
  Constant(bool),
  Literal(Literal),
  Not(Box<Self>),
  And(Vec<Self>),
  Or(Vec<Self>),
}

impl BooleanFormula {
  pub fn conjunction(terms: impl IntoIterator<Item = Self>) -> Self {
    Self::combine(terms, true, Self::And)
  }

  pub fn disjunction(terms: impl IntoIterator<Item = Self>) -> Self {
    Self::combine(terms, false, Self::Or)
  }

  fn combine(terms: impl IntoIterator<Item = Self>, identity: bool, combine: impl FnOnce(Vec<Self>) -> Self) -> Self {
    let mut terms = terms.into_iter();
    let Some(first) = terms.next() else {
      return Self::Constant(identity);
    };
    let Some(second) = terms.next() else {
      return first;
    };
    combine([first, second].into_iter().chain(terms).collect())
  }

  /// Collect the distinct fact IDs in ascending order, suitable for assigning dense variable labels.
  pub fn fact_ids(&self) -> Vec<usize> {
    let mut facts = Vec::new();
    self.collect_facts(&mut facts);
    facts.sort_unstable();
    facts.dedup();
    facts
  }

  fn collect_facts(&self, facts: &mut Vec<usize>) {
    match self {
      Self::Constant(_) => {}
      Self::Literal(lit) => facts.push(lit.fact_id()),
      Self::Not(term) => term.collect_facts(facts),
      Self::And(terms) | Self::Or(terms) => {
        for term in terms {
          term.collect_facts(facts);
        }
      }
    }
  }
}

/// Weighted-counting operations. Elements may own data, such as sparse gradients.
pub trait Semiring {
  type Element: Clone;

  fn zero(&self) -> Self::Element;
  fn one(&self) -> Self::Element;
  fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element;
  fn mult(&self, a: Self::Element, b: Self::Element) -> Self::Element;
  /// Return the complementary weight, so `add(a, negate(a))` is one.
  fn negate(&self, a: Self::Element) -> Self::Element;
}
