use std::collections::BTreeSet;
pub use std::iter::FromIterator;

use indexmap::IndexMap;

#[derive(Clone, Debug, Default)]
pub struct Disjunction {
  pub facts: BTreeSet<usize>,
}

impl Disjunction {
  pub fn add_fact_id(&mut self, fact_id: usize) {
    self.facts.insert(fact_id);
  }

  /// Note: Assumes that the #facts > 2
  pub fn has_conflict(&self, facts: &BTreeSet<usize>) -> bool {
    // Short cut 1
    if self.facts.len() < 2 {
      return false;
    }

    // Short cut 2
    let j_last = self.facts.last().unwrap(); // Note: disj.len >= 2
    let j_first = self.facts.first().unwrap(); // Note: disj.len >= 2
    let f_last = facts.last().unwrap(); // Note: facts.len >= 2
    let f_first = facts.first().unwrap(); // Note: facts.len >= 2
    if j_last < f_first || f_last < j_first {
      return false;
    }

    // Start iteration
    let mut j_iter = self.facts.iter();
    let mut f_iter = facts.iter();
    let mut has_same = false;
    let mut j_curr = j_iter.next();
    let mut f_curr = f_iter.next();
    while let (Some(j_elem), Some(f_elem)) = (j_curr, f_curr) {
      if j_elem == f_elem {
        if has_same {
          return true;
        }
        has_same = true;
        j_curr = j_iter.next();
        f_curr = f_iter.next();
      } else if j_elem < f_elem {
        j_curr = j_iter.next();
      } else {
        f_curr = f_iter.next();
      }
    }
    false
  }
}

impl FromIterator<usize> for Disjunction {
  fn from_iter<T>(iter: T) -> Self
  where
    T: IntoIterator<Item = usize>,
  {
    Self {
      facts: iter.into_iter().collect(),
    }
  }
}

#[derive(Clone, Debug, Default)]
pub struct Disjunctions {
  groups: IndexMap<usize, Disjunction>,
}

impl Disjunctions {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn is_empty(&self) -> bool {
    self.groups.is_empty()
  }

  pub fn len(&self) -> usize {
    self.groups.len()
  }

  pub fn iter(&self) -> impl Iterator<Item = &Disjunction> {
    self.groups.values()
  }

  pub fn has_conflict(&self, facts: &BTreeSet<usize>) -> bool {
    facts.len() >= 2 && self.iter().any(|group| group.has_conflict(facts))
  }

  pub fn add_disjunction(&mut self, disj_id: usize, fact_id: usize) {
    self.groups.entry(disj_id).or_default().add_fact_id(fact_id);
  }
}
