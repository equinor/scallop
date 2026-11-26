use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;

use num_bigint::BigUint;

use super::{BooleanFormula, Literal, Semiring, WmcBackend, WmcModel};

/// A nonnegative exact weight rounded down, up, and to nearest (ties to even).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbabilityWeight {
  pub lower: f64,
  pub upper: f64,
  pub nearest: f64,
}

impl ProbabilityWeight {
  fn from_ratio(numerator: &BigUint, denominator: &BigUint) -> Self {
    let zero = BigUint::default();
    if numerator == &zero {
      return Self {
        lower: 0.0,
        upper: 0.0,
        nearest: 0.0,
      };
    }

    debug_assert_ne!(denominator, &zero);
    let mut exponent = numerator.bits() as i64 - denominator.bits() as i64;
    let below_power = if exponent >= 0 {
      numerator < &(denominator << exponent as usize)
    } else {
      &(numerator << (-exponent) as usize) < denominator
    };
    exponent -= i64::from(below_power);
    if exponent > 1023 {
      return Self {
        lower: f64::MAX,
        upper: f64::INFINITY,
        nearest: f64::INFINITY,
      };
    }

    // Divide at the precision of the binary64 binade, or at the subnormal step.
    let shift = 52 - exponent.max(-1022);
    let (scaled_numerator, scaled_denominator) = if shift >= 0 {
      (numerator << shift as usize, denominator.clone())
    } else {
      (numerator.clone(), denominator << (-shift) as usize)
    };
    let significand = &scaled_numerator / &scaled_denominator;
    let remainder = scaled_numerator - &significand * &scaled_denominator;
    let bits = u64::try_from(significand).expect("binary64 significand fits in 53 bits")
      + (((exponent + 1022).max(0) as u64) << 52);

    let upper_bits = bits + u64::from(remainder != zero);
    let nearest_bits = match (remainder << 1usize).cmp(&scaled_denominator) {
      Ordering::Greater => upper_bits,
      Ordering::Equal if bits & 1 != 0 => upper_bits,
      _ => bits,
    };
    Self {
      lower: f64::from_bits(bits),
      upper: f64::from_bits(upper_bits),
      nearest: f64::from_bits(nearest_bits),
    }
  }
}

/// An invalid categorical distribution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CategoricalError {
  DuplicateOutcome { fact_id: usize },
  InvalidWeight { fact_id: usize },
  TotalWeightExceedsOne,
  ZeroTotalWeight,
}

impl fmt::Display for CategoricalError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::DuplicateOutcome { fact_id } => write!(f, "duplicate categorical outcome with fact ID {fact_id}"),
      Self::InvalidWeight { fact_id } => write!(f, "categorical outcome {fact_id} must have a finite weight in [0, 1]"),
      Self::TotalWeightExceedsOne => write!(f, "exact total categorical weight exceeds one"),
      Self::ZeroTotalWeight => write!(f, "cannot normalize a categorical group with zero total weight"),
    }
  }
}

impl std::error::Error for CategoricalError {}

#[derive(Clone, Copy, Debug)]
enum Selection {
  Impossible,
  FirstSuccess(usize),
  AllFail,
}

/// Mutually exclusive alternatives over original fact IDs.
#[derive(Clone, Debug)]
pub struct CategoricalGroup {
  outcome_ids: Vec<usize>,
  selections: Vec<Selection>,
  selector_weights: Vec<ProbabilityWeight>,
  total_weight: ProbabilityWeight,
}

impl CategoricalGroup {
  /// With `normalize` disabled, the unassigned mass goes to the world in which
  /// none of the outcomes occur. With it enabled, divide by the exact total and
  /// choose exactly one outcome. Validation uses the exact values represented
  /// by the input floats, not a floating-point sum.
  pub fn new(outcomes: Vec<(usize, f64)>, normalize: bool) -> Result<Self, CategoricalError> {
    let mut seen = HashSet::with_capacity(outcomes.len());
    let mut outcome_ids = Vec::with_capacity(outcomes.len());
    let mut weights = Vec::with_capacity(outcomes.len());
    let mut total = BigUint::default();
    for (fact_id, weight) in outcomes {
      if !seen.insert(fact_id) {
        return Err(CategoricalError::DuplicateOutcome { fact_id });
      }
      if !(0.0..=1.0).contains(&weight) {
        return Err(CategoricalError::InvalidWeight { fact_id });
      }
      let weight = binary64_units(weight);
      total += &weight;
      outcome_ids.push(fact_id);
      weights.push(weight);
    }

    let one = BigUint::from(1u8) << 1074usize;
    let zero = BigUint::default();
    let (mut remaining, residual) = if normalize {
      if total == zero {
        return Err(CategoricalError::ZeroTotalWeight);
      }
      (total.clone(), BigUint::default())
    } else {
      if total > one {
        return Err(CategoricalError::TotalWeightExceedsOne);
      }
      (one.clone(), &one - &total)
    };

    // Leave the largest mass implicit. Each explicit selector is then at most
    // 1/2, so a rounded near-one weight cannot erase a small alternative.
    let mut largest = &residual;
    let mut implicit = None;
    for (index, weight) in weights.iter().enumerate() {
      if weight > largest {
        largest = weight;
        implicit = Some(index);
      }
    }

    let mut selector_weights = Vec::new();
    let mut selections = Vec::with_capacity(weights.len());
    for (index, weight) in weights.into_iter().enumerate() {
      let selection = if weight == zero {
        Selection::Impossible
      } else if implicit == Some(index) {
        Selection::AllFail
      } else {
        let selection = Selection::FirstSuccess(selector_weights.len());
        selector_weights.push(ProbabilityWeight::from_ratio(&weight, &remaining));
        remaining -= weight;
        selection
      };
      selections.push(selection);
    }
    if implicit.is_some() && residual != zero {
      selector_weights.push(ProbabilityWeight::from_ratio(&residual, &remaining));
    }

    Ok(Self {
      outcome_ids,
      selections,
      selector_weights,
      total_weight: ProbabilityWeight::from_ratio(&total, &one),
    })
  }

  /// Original fact IDs, in constructor order, including zero-weight outcomes.
  pub fn outcome_ids(&self) -> &[usize] {
    &self.outcome_ids
  }

  /// The original, unnormalized sum, which may exceed one.
  pub fn total_weight(&self) -> ProbabilityWeight {
    self.total_weight
  }
}

// Convert a validated binary64 to integer multiples of the smallest subnormal.
fn binary64_units(weight: f64) -> BigUint {
  let bits = weight.to_bits();
  let exponent = ((bits >> 52) & 0x7ff) as usize;
  let fraction = bits & ((1u64 << 52) - 1);
  if exponent == 0 {
    BigUint::from(fraction)
  } else {
    BigUint::from(fraction | (1u64 << 52)) << (exponent - 1)
  }
}

/// A distribution with independent categorical groups and independent Bernoulli
/// facts. Both kinds of weights are supplied to the backend as Boolean weights.
pub struct CategoricalModel<'a, I, W> {
  groups: &'a [CategoricalGroup],
  locations: HashMap<usize, (usize, usize)>,
  selector_offsets: Vec<usize>,
  forced: HashMap<usize, Option<usize>>,
  independent: I,
  selector_weight: W,
}

impl<'a, I, W> CategoricalModel<'a, I, W> {
  /// The independent callback receives original fact IDs; the selector callback
  /// converts enclosed Bernoulli weights to the counting algebra.
  ///
  /// Panics if a fact ID occurs in more than one group.
  pub fn new<E>(groups: &'a [CategoricalGroup], independent: I, selector_weight: W) -> Self
  where
    I: Fn(&usize) -> E,
    W: Fn(ProbabilityWeight) -> E,
  {
    let mut locations = HashMap::new();
    let mut selector_offsets = Vec::with_capacity(groups.len() + 1);
    selector_offsets.push(0);
    for (group_index, group) in groups.iter().enumerate() {
      for (outcome_index, &fact_id) in group.outcome_ids().iter().enumerate() {
        assert!(
          locations.insert(fact_id, (group_index, outcome_index)).is_none(),
          "categorical fact ID {fact_id} occurs in multiple groups"
        );
      }
      selector_offsets.push(selector_offsets[group_index] + group.selector_weights.len());
    }
    Self {
      groups,
      locations,
      selector_offsets,
      forced: HashMap::new(),
      independent,
      selector_weight,
    }
  }

  /// Replace one group's distribution with a single world, even if that world
  /// has zero original weight. `None` forces all outcomes false; `Some` selects
  /// an original fact ID. Later calls override earlier calls for the same group.
  ///
  /// Panics if the group index or selected outcome does not exist.
  pub fn force(mut self, group_index: usize, outcome: Option<usize>) -> Self {
    assert!(group_index < self.groups.len(), "invalid categorical group index");
    if let Some(fact_id) = outcome {
      assert!(
        self
          .locations
          .get(&fact_id)
          .is_some_and(|&(group, _)| group == group_index),
        "categorical fact ID {fact_id} is not in group {group_index}"
      );
    }
    self.forced.insert(group_index, outcome);
    self
  }

  fn lower_outcome(&self, fact_id: usize, group: usize, outcome: usize) -> BooleanFormula {
    if let Some(forced_outcome) = self.forced.get(&group) {
      return BooleanFormula::Constant(*forced_outcome == Some(fact_id));
    }
    let base = self.selector_offsets[group];
    let (end, selected) = match self.groups[group].selections[outcome] {
      Selection::Impossible => return BooleanFormula::Constant(false),
      Selection::FirstSuccess(index) => (base + index, Some(BooleanFormula::Literal(Literal::Pos(base + index)))),
      Selection::AllFail => (self.selector_offsets[group + 1], None),
    };
    BooleanFormula::conjunction(
      (base..end)
        .map(|index| BooleanFormula::Literal(Literal::Neg(index)))
        .chain(selected),
    )
  }
}

impl<S, I, W> WmcModel<S> for CategoricalModel<'_, I, W>
where
  S: Semiring,
  I: Fn(&usize) -> S::Element,
  W: Fn(ProbabilityWeight) -> S::Element,
{
  fn count(&self, event: &BooleanFormula, algebra: &S, backend: &impl WmcBackend) -> S::Element {
    if self.groups.is_empty() {
      return backend.wmc(event, algebra, &self.independent);
    }

    let mut independent_ids = Vec::new();
    let independent_base = self.selector_offsets[self.groups.len()];
    let replacements = event
      .fact_ids()
      .into_iter()
      .map(|fact_id| {
        let outcome = if let Some(&(group, outcome)) = self.locations.get(&fact_id) {
          self.lower_outcome(fact_id, group, outcome)
        } else {
          let label = independent_base + independent_ids.len();
          independent_ids.push(fact_id);
          BooleanFormula::Literal(Literal::Pos(label))
        };
        (fact_id, outcome)
      })
      .collect();
    let lowered = lower(event, &replacements);
    backend.wmc(&lowered, algebra, &|label| {
      if *label < independent_base {
        let group = self.selector_offsets.partition_point(|offset| offset <= label) - 1;
        let index = label - self.selector_offsets[group];
        (self.selector_weight)(self.groups[group].selector_weights[index])
      } else {
        (self.independent)(&independent_ids[label - independent_base])
      }
    })
  }
}

fn lower(event: &BooleanFormula, replacements: &HashMap<usize, BooleanFormula>) -> BooleanFormula {
  match event {
    BooleanFormula::Constant(value) => BooleanFormula::Constant(*value),
    BooleanFormula::Literal(literal) => {
      let outcome = replacements[&literal.fact_id()].clone();
      if literal.sign() {
        outcome
      } else {
        negate(outcome)
      }
    }
    BooleanFormula::Not(term) => negate(lower(term, replacements)),
    BooleanFormula::And(terms) => BooleanFormula::conjunction(terms.iter().map(|term| lower(term, replacements))),
    BooleanFormula::Or(terms) => BooleanFormula::disjunction(terms.iter().map(|term| lower(term, replacements))),
  }
}

fn negate(event: BooleanFormula) -> BooleanFormula {
  match event {
    BooleanFormula::Constant(value) => BooleanFormula::Constant(!value),
    BooleanFormula::Literal(lit) => BooleanFormula::Literal(lit.negate()),
    BooleanFormula::Not(term) => *term,
    other => BooleanFormula::Not(Box::new(other)),
  }
}

#[cfg(test)]
mod tests {
  use super::super::{count, DefaultWmcBackend, RealSemiring};
  use super::*;

  fn pos(id: usize) -> BooleanFormula {
    BooleanFormula::Literal(Literal::Pos(id))
  }

  fn neg(id: usize) -> BooleanFormula {
    BooleanFormula::Literal(Literal::Neg(id))
  }

  #[test]
  fn categorical_weights_validate_the_exact_total() {
    let tiny = f64::from_bits(1);
    assert_eq!(
      CategoricalGroup::new(vec![(0, 1.0), (1, tiny)], false).unwrap_err(),
      CategoricalError::TotalWeightExceedsOne,
    );
    assert_eq!(
      CategoricalGroup::new(vec![(0, -0.0), (1, 0.0)], true).unwrap_err(),
      CategoricalError::ZeroTotalWeight,
    );
    assert_eq!(
      CategoricalGroup::new(vec![(0, 0.0), (0, 0.5)], false).unwrap_err(),
      CategoricalError::DuplicateOutcome { fact_id: 0 },
    );
    for invalid in [f64::NAN, f64::INFINITY, -tiny, 1.0f64.next_up()] {
      assert_eq!(
        CategoricalGroup::new(vec![(1, invalid)], true).unwrap_err(),
        CategoricalError::InvalidWeight { fact_id: 1 },
      );
    }

    let halfway = f64::EPSILON / 2.0;
    for (small, lower, upper, nearest) in [
      (halfway, 1.0, 1.0f64.next_up(), 1.0),
      (
        3.0 * halfway,
        1.0f64.next_up(),
        1.0f64.next_up().next_up(),
        1.0f64.next_up().next_up(),
      ),
    ] {
      let group = CategoricalGroup::new(vec![(0, 1.0), (usize::MAX, small)], true).unwrap();
      assert_eq!(group.outcome_ids(), &[0, usize::MAX]);
      assert_eq!(group.total_weight(), ProbabilityWeight { lower, upper, nearest });
    }
  }

  #[test]
  fn categorical_count_mixes_sparse_groups_and_independent_facts() {
    let groups = [
      CategoricalGroup::new(vec![(0, 3.0 / 8.0), (usize::MAX, 1.0 / 8.0)], false).unwrap(),
      CategoricalGroup::new(vec![(7, 1.0), (70, 1.0), (700, 0.0)], true).unwrap(),
    ];
    let model = CategoricalModel::new(
      &groups,
      |id| {
        assert_eq!(*id, 1);
        0.25
      },
      |weight| weight.nearest,
    );
    let event = BooleanFormula::disjunction([
      BooleanFormula::conjunction([pos(0), neg(usize::MAX), pos(7)]),
      BooleanFormula::conjunction([neg(0), neg(usize::MAX), pos(1)]),
    ]);
    assert_eq!(count(&event, &model, &RealSemiring, &DefaultWmcBackend), 5.0 / 16.0);
    let impossible = BooleanFormula::conjunction([pos(0), pos(usize::MAX)]);
    assert_eq!(count(&impossible, &model, &RealSemiring, &DefaultWmcBackend), 0.0);
    let residual = BooleanFormula::Not(Box::new(BooleanFormula::disjunction([pos(7), pos(70), pos(700)])));
    assert_eq!(count(&residual, &model, &RealSemiring, &DefaultWmcBackend), 0.0);
  }

  #[test]
  fn categorical_count_preserves_a_subnormal_alternative_to_a_large_outcome() {
    let tiny = f64::from_bits(1);
    let groups = [CategoricalGroup::new(vec![(7, 1.0), (8, tiny)], true).unwrap()];
    let model = CategoricalModel::new(
      &groups,
      |_| panic!("unexpected independent fact"),
      |weight| {
        assert_eq!(
          weight,
          ProbabilityWeight {
            lower: 0.0,
            upper: tiny,
            nearest: tiny,
          },
        );
        weight.nearest
      },
    );
    assert_eq!(count(&pos(8), &model, &RealSemiring, &DefaultWmcBackend), tiny);
  }

  #[test]
  fn categorical_count_can_force_zero_weight_worlds() {
    let groups = [CategoricalGroup::new(vec![(7, 0.0), (8, 1.0)], false).unwrap()];
    let model = || {
      CategoricalModel::new(
        &groups,
        |_| panic!("unexpected independent fact"),
        |_| panic!("unexpected selector"),
      )
    };
    let event = BooleanFormula::conjunction([pos(7), neg(8)]);
    assert_eq!(count(&event, &model(), &RealSemiring, &DefaultWmcBackend), 0.0);
    assert_eq!(
      count(&event, &model().force(0, Some(7)), &RealSemiring, &DefaultWmcBackend),
      1.0,
    );
    let residual = BooleanFormula::conjunction([neg(7), neg(8)]);
    assert_eq!(
      count(&residual, &model().force(0, None), &RealSemiring, &DefaultWmcBackend),
      1.0,
    );
  }
}
