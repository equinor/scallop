use std::collections::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BooleanFormula {
  False,
  True,
  Pos {
    var_id: usize,
  },
  Neg {
    var_id: usize,
  },
  Not {
    form: Box<BooleanFormula>,
  },
  And {
    left: Box<BooleanFormula>,
    right: Box<BooleanFormula>,
  },
  Or {
    left: Box<BooleanFormula>,
    right: Box<BooleanFormula>,
  },
}

/// Create a boolean formula literal using var_id
pub fn bf(var_id: usize) -> BooleanFormula {
  BooleanFormula::Pos { var_id }
}

/// Create a boolean formula literal using var_id
pub fn bf_pos(var_id: usize) -> BooleanFormula {
  BooleanFormula::Pos { var_id }
}

/// Create a boolean formula literal using var_id
pub fn bf_neg(var_id: usize) -> BooleanFormula {
  BooleanFormula::Neg { var_id }
}

/// Create a boolean formula true
pub fn bf_true() -> BooleanFormula {
  BooleanFormula::True
}

/// Create a boolean formula false
pub fn bf_false() -> BooleanFormula {
  BooleanFormula::False
}

/// Create a conjunction formula over bs
pub fn bf_conjunction<I>(mut bs: I) -> BooleanFormula
where
  I: Iterator<Item = BooleanFormula>,
{
  if let Some(mut agg) = bs.next() {
    for next_fact in bs {
      agg = agg & next_fact;
    }
    agg
  } else {
    bf_true()
  }
}

/// Create a disjunction formula over bs
pub fn bf_disjunction<I>(mut bs: I) -> BooleanFormula
where
  I: Iterator<Item = BooleanFormula>,
{
  if let Some(mut agg) = bs.next() {
    for next_fact in bs {
      agg = agg | next_fact;
    }
    agg
  } else {
    bf_false()
  }
}

impl std::ops::BitAnd for BooleanFormula {
  type Output = Self;

  fn bitand(self, rhs: Self) -> Self {
    Self::And {
      left: Box::new(self),
      right: Box::new(rhs),
    }
  }
}

impl std::ops::BitOr for BooleanFormula {
  type Output = Self;

  fn bitor(self, rhs: Self) -> Self {
    Self::Or {
      left: Box::new(self),
      right: Box::new(rhs),
    }
  }
}

impl std::ops::Not for BooleanFormula {
  type Output = Self;

  fn not(self) -> Self {
    match self {
      Self::Pos { var_id } => Self::Neg { var_id },
      other => Self::Not { form: Box::new(other) },
    }
  }
}

impl std::fmt::Display for BooleanFormula {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::And { left, right } => write!(f, "a{left}{right}"),
      Self::Or { left, right } => write!(f, "o{left}{right}"),
      Self::Not { form } => write!(f, "n{form}"),
      Self::Neg { var_id } => write!(f, "n{var_id}"),
      Self::Pos { var_id } => write!(f, "p{var_id}"),
      Self::False => f.write_str("f"),
      Self::True => f.write_str("t"),
    }
  }
}

impl BooleanFormula {
  pub fn collect_vars(&self) -> Vec<usize> {
    let mut set = BTreeSet::new();
    self.collect_vars_helper(&mut set);
    set.into_iter().collect()
  }

  pub fn collect_sorted_vars(&self) -> Vec<usize> {
    let mut occurrence = HashMap::new();
    self.count_var_occurrence_helper(&mut occurrence);
    let mut vars = self.collect_vars();
    vars.sort_by_key(|v| -(occurrence[v] as i32));
    vars
  }

  fn count_var_occurrence_helper(&self, occurrence: &mut HashMap<usize, usize>) {
    match self {
      Self::True | Self::False => {}
      Self::Pos { var_id } => {
        *occurrence.entry(*var_id).or_default() += 1;
      }
      Self::Neg { var_id } => {
        *occurrence.entry(*var_id).or_default() += 1;
      }
      Self::Not { form } => {
        form.count_var_occurrence_helper(occurrence);
      }
      Self::And { left, right } => {
        left.count_var_occurrence_helper(occurrence);
        right.count_var_occurrence_helper(occurrence);
      }
      Self::Or { left, right } => {
        left.count_var_occurrence_helper(occurrence);
        right.count_var_occurrence_helper(occurrence);
      }
    }
  }

  fn collect_vars_helper(&self, collection: &mut BTreeSet<usize>) {
    match self {
      Self::True | Self::False => {}
      Self::Pos { var_id } => {
        collection.insert(*var_id);
      }
      Self::Neg { var_id } => {
        collection.insert(*var_id);
      }
      Self::Not { form } => {
        form.collect_vars_helper(collection);
      }
      Self::And { left, right } => {
        left.collect_vars_helper(collection);
        right.collect_vars_helper(collection);
      }
      Self::Or { left, right } => {
        left.collect_vars_helper(collection);
        right.collect_vars_helper(collection);
      }
    }
  }
}
