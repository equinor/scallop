use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use itertools::Itertools;
use petgraph::visit::{Dfs, Reversed};
use thiserror::Error;

use crate::common::binary_op::BinaryOp;
use crate::common::expr::Expr;
use crate::common::foreign_aggregate::Aggregator;
use crate::common::foreign_aggregates::{CountAggregator, ExistsAggregator, SumProdAggregator};
use crate::common::foreign_predicate::ForeignPredicate;
use crate::common::input_tag::DynamicInputTag;
use crate::common::tuple::Tuple;
use crate::common::tuple_type::TupleType;
use crate::common::unary_op::UnaryOp;
use crate::common::value_type::ValueType;
use crate::compiler::ram::{Dataflow, Fact, Program};
use crate::runtime::database::extensional::parse_input_file_facts;
use crate::runtime::database::io::InputFileRegistry;
use crate::runtime::database::{DatabaseError, StorageMetadata};
use crate::runtime::dynamic::{DynamicElement, DynamicElements, DynamicExecutionContext, ExecutionOptions};
use crate::runtime::env::{RuntimeEnvironment, Scheduler};
use crate::runtime::error::{check_interruption, RuntimeError};
use crate::runtime::provenance::{
  count, with_prepared_events, AsBooleanFormula, BooleanFormula, CategoricalError, CategoricalGroup, CategoricalModel,
  Clause, DNFFormula, DefaultWmcBackend, DualNumberSemiring, PreparedWmc, ProbabilityWeight, Provenance, RealSemiring,
  Semiring, StaticInputTag, Tag, WmcModel,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbabilityInterval {
  pub lower: f64,
  pub upper: f64,
}

impl ProbabilityInterval {
  const ZERO: Self = Self { lower: 0.0, upper: 0.0 };
  const ONE: Self = Self { lower: 1.0, upper: 1.0 };

  fn point(p: f64) -> Self {
    Self { lower: p, upper: p }
  }

  fn intersect(self, other: Self) -> Self {
    Self {
      lower: self.lower.max(other.lower),
      upper: self.upper.min(other.upper),
    }
  }

  fn within(self, tolerance: f64) -> bool {
    self.lower == self.upper || tolerance == 1.0 || (self.upper - self.lower).next_up() <= tolerance
  }
}

impl fmt::Display for ProbabilityInterval {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}, {}]", self.lower, self.upper)
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundsStop {
  Tolerance,
  FactLimit,
  TimeLimit,
  Complete,
}

#[derive(Clone, Debug)]
pub struct BoundsResult {
  pub interval: ProbabilityInterval,
  pub active_facts: usize,
  pub stop_reason: BoundsStop,
  pub gradients: Vec<GradientBound>,
  pub exact_probability: Option<f64>,
  /// Nonzero upper-triangular entries, indexed in `gradients` order.
  pub hessian: Option<Vec<(usize, usize, f64)>>,
}

#[derive(Clone)]
pub struct BoundsInput {
  pub rows: BoundsRows,
  pub probability: f64,
  pub group: Option<BoundsInputGroup>,
  pub differentiate: bool,
}

#[derive(Clone)]
pub enum BoundsRows {
  Fact((String, Tuple)),
  Joint(Vec<(String, Tuple)>),
}

impl BoundsRows {
  fn as_slice(&self) -> &[(String, Tuple)] {
    match self {
      Self::Fact(row) => std::slice::from_ref(row),
      Self::Joint(rows) => rows,
    }
  }

  fn into_rows(self) -> impl Iterator<Item = (String, Tuple)> {
    match self {
      Self::Fact(row) => itertools::Either::Left(std::iter::once(row)),
      Self::Joint(rows) => itertools::Either::Right(rows.into_iter()),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundsInputGroup {
  Categorical(usize),
  Joint(usize),
}

#[derive(Clone, Copy)]
pub struct BoundsOptions {
  pub tolerance: Option<f64>,
  pub max_facts: usize,
  pub normalize_disjunctions: bool,
  pub fact_order: FactOrder,
  pub checkpoint_schedule: CheckpointSchedule,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FactOrder {
  Input,
  Weight,
  Required,
  Lookahead,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CheckpointSchedule {
  Doubling,
  Adaptive,
}

#[derive(Clone, Copy, Debug)]
pub struct GradientBound {
  pub input: usize,
  pub lower: f64,
  pub upper: f64,
  pub exact: Option<f64>,
}

#[derive(Clone, Copy)]
pub struct GradientTarget {
  /// Position of the trainable weight in the input snapshot.
  pub input: usize,
  pub tolerance: f64,
}

#[derive(Debug, Error)]
pub enum BoundsError {
  #[error("bounds tolerance must be between zero and one")]
  InvalidTolerance,
  #[error("a resumed bounds query cannot reduce its fact budget")]
  DecreasedFactBudget,
  #[error("resolved query derivatives exceed the finite binary64 range")]
  NonFiniteDerivative,
  #[error("gradient tolerance must be finite and non-negative, got {0}")]
  InvalidGradientTolerance(f64),
  #[error("input {0} does not identify a differentiable weight")]
  UnknownGradientInput(usize),
  #[error("more than one gradient tolerance for input {0}")]
  DuplicateGradientTarget(usize),
  #[error("provide at least one gradient tolerance or a probability tolerance")]
  NoToleranceTarget,
  #[error("invalid probability {0}; expected a finite number between zero and one")]
  InvalidProbability(f64),
  #[error("unknown relation `{0}` in the compiled program")]
  UnknownRelation(String),
  #[error("tuple does not match the type of relation `{0}`")]
  InvalidTuple(String),
  #[error("anytime bounds do not support {0}")]
  Unsupported(&'static str),
  #[error("anytime bounds do not support foreign {kind} `{name}`: {reason}")]
  UnsupportedForeignCall {
    kind: &'static str,
    name: String,
    reason: &'static str,
  },
  #[error("a refinement checkpoint found a rule choice outside the discovered support")]
  UndiscoveredChoice,
  #[error(transparent)]
  Database(#[from] DatabaseError),
  #[error(transparent)]
  Runtime(#[from] RuntimeError),
  #[error(transparent)]
  Categorical(#[from] CategoricalError),
}

/// Bound the full probability of one ground query in a finite stratified program.
///
/// Each checkpoint makes a larger prefix of the uncertain facts symbolic.
/// The remaining facts start with the interval [false, true]. After both
/// formulas reach a fixed point, they can be counted. Without a tolerance,
/// evaluate through the fact budget before computing requested gradients.
pub fn evaluate(
  program: &Program,
  inputs: impl IntoIterator<Item = BoundsInput>,
  relation: &str,
  query: &Tuple,
  options: BoundsOptions,
) -> Result<BoundsResult, BoundsError> {
  evaluate_ground(program, inputs, relation, query, options, GroundGoal::default())
}

/// Evaluate a ground query's gradients and, when resolved, its Hessian.
pub fn evaluate_with_hessian(
  program: &Program,
  inputs: impl IntoIterator<Item = BoundsInput>,
  relation: &str,
  query: &Tuple,
  options: BoundsOptions,
) -> Result<BoundsResult, BoundsError> {
  if options.tolerance.is_some() {
    return Err(BoundsError::Unsupported(
      "second derivatives with probability-only tolerance stopping",
    ));
  }
  evaluate_ground(
    program,
    inputs,
    relation,
    query,
    options,
    GroundGoal {
      hessian: true,
      gradients: None,
    },
  )
}

/// Evaluate through the first checkpoint meeting all requested gradient and probability tolerances.
pub fn evaluate_until_gradients(
  program: &Program,
  inputs: impl IntoIterator<Item = BoundsInput>,
  relation: &str,
  query: &Tuple,
  options: BoundsOptions,
  targets: &[GradientTarget],
  hessian: bool,
) -> Result<BoundsResult, BoundsError> {
  if targets.is_empty() && options.tolerance.is_none() {
    return Err(BoundsError::NoToleranceTarget);
  }
  evaluate_ground(
    program,
    inputs,
    relation,
    query,
    options,
    GroundGoal {
      hessian,
      gradients: Some(targets),
    },
  )
}

#[derive(Default)]
struct GroundGoal<'a> {
  hessian: bool,
  gradients: Option<&'a [GradientTarget]>,
}

fn evaluate_ground(
  program: &Program,
  inputs: impl IntoIterator<Item = BoundsInput>,
  relation: &str,
  query: &Tuple,
  options: BoundsOptions,
  goal: GroundGoal<'_>,
) -> Result<BoundsResult, BoundsError> {
  validate_ground_query(program, relation, query, options)?;
  let prepared = prepare_relation(program, inputs, relation, options.normalize_disjunctions)?;
  GroundRefinement::default().advance(&prepared, relation, query, options, None, goal)
}

/// Evaluate ground queries that all see the same input snapshot.
pub fn evaluate_batch(
  program: &Program,
  inputs: &[BoundsInput],
  queries: &[(String, Tuple)],
  options: BoundsOptions,
) -> Result<Vec<BoundsResult>, BoundsError> {
  let mut remaining = queries.iter().map(|(relation, _)| relation.as_str()).counts();
  let mut prepared = HashMap::new();
  let mut first_result = HashMap::<(&str, &Tuple), usize>::new();
  let mut results: Vec<BoundsResult> = Vec::with_capacity(queries.len());
  for (relation, tuple) in queries {
    let relation = relation.as_str();
    let pending = remaining
      .get_mut(relation)
      .expect("query relations are counted ahead of time");
    *pending -= 1;
    let result = if let Some(&index) = first_result.get(&(relation, tuple)) {
      results[index].clone()
    } else {
      validate_ground_query(program, relation, tuple, options)?;
      let result = if let Some(snapshot) = prepared.get(relation) {
        evaluate_prepared(snapshot, relation, tuple, options)?
      } else {
        let snapshot = prepare_relation(
          program,
          inputs.iter().cloned(),
          relation,
          options.normalize_disjunctions,
        )?;
        let result = evaluate_prepared(&snapshot, relation, tuple, options)?;
        if *pending > 0 {
          prepared.insert(relation, snapshot);
        }
        result
      };
      first_result.insert((relation, tuple), results.len());
      result
    };
    results.push(result);
    if *pending == 0 {
      prepared.remove(relation);
    }
  }
  Ok(results)
}

fn validate_ground_query(
  program: &Program,
  relation: &str,
  query: &Tuple,
  options: BoundsOptions,
) -> Result<(), BoundsError> {
  if options.tolerance.is_some_and(|t| !(0.0..=1.0).contains(&t)) {
    return Err(BoundsError::InvalidTolerance);
  }
  let relation_type = program
    .relation_tuple_type(relation)
    .ok_or_else(|| BoundsError::UnknownRelation(relation.into()))?;
  if !relation_type.matches(query) {
    return Err(BoundsError::InvalidTuple(relation.into()));
  }
  Ok(())
}

fn evaluate_prepared(
  prepared: &PreparedRelation,
  relation: &str,
  query: &Tuple,
  options: BoundsOptions,
) -> Result<BoundsResult, BoundsError> {
  GroundRefinement::default().advance(prepared, relation, query, options, None, GroundGoal::default())
}

/// A resumable ground probability query over an owned, immutable input snapshot.
pub struct BoundsSession {
  prepared: PreparedRelation,
  relation: String,
  query: Tuple,
  options: BoundsOptions,
  refinement: GroundRefinement,
}

impl BoundsSession {
  pub fn new(
    program: &Program,
    inputs: impl IntoIterator<Item = BoundsInput>,
    (relation, query): (String, Tuple),
    tolerance: f64,
    normalize_disjunctions: bool,
    fact_order: FactOrder,
    checkpoint_schedule: CheckpointSchedule,
  ) -> Result<Self, BoundsError> {
    let options = BoundsOptions {
      tolerance: Some(tolerance),
      max_facts: 0,
      normalize_disjunctions,
      fact_order,
      checkpoint_schedule,
    };
    validate_ground_query(program, &relation, &query, options)?;
    let prepared = prepare_relation(program, inputs, &relation, normalize_disjunctions)?;
    Ok(Self {
      prepared,
      relation,
      query,
      options,
      refinement: GroundRefinement::default(),
    })
  }

  pub fn advance(&mut self, max_facts: usize) -> Result<BoundsResult, BoundsError> {
    self.advance_inner(max_facts, None)
  }

  /// Stop cooperatively when `expired` returns true; an unfinished checkpoint is discarded.
  pub fn advance_until(&mut self, max_facts: usize, expired: impl Fn() -> bool) -> Result<BoundsResult, BoundsError> {
    self.advance_inner(max_facts, Some(&expired))
  }

  fn advance_inner(
    &mut self,
    max_facts: usize,
    expired: Option<&dyn Fn() -> bool>,
  ) -> Result<BoundsResult, BoundsError> {
    if max_facts < self.options.max_facts {
      return Err(BoundsError::DecreasedFactBudget);
    }
    self.options.max_facts = max_facts;
    match self.refinement.advance(
      &self.prepared,
      &self.relation,
      &self.query,
      self.options,
      expired,
      GroundGoal::default(),
    ) {
      Err(BoundsError::Runtime(RuntimeError::Interrupted)) => {
        let active = self.refinement.active.unwrap_or(0);
        Ok(
          self
            .refinement
            .certificate(self.prepared.model.costs[active], BoundsStop::TimeLimit),
        )
      }
      result => result,
    }
  }
}

struct GroundRefinement {
  activation: Option<Vec<usize>>,
  lookahead: Option<(usize, Vec<usize>)>,
  active: Option<usize>,
  next: usize,
  interval: ProbabilityInterval,
}

struct GradientConvergence {
  selected: Vec<(usize, Option<usize>)>,
  tracked: Vec<(usize, f64, ProbabilityInterval)>,
}

impl GradientConvergence {
  fn new(targets: &[GradientTarget], differentiated: &[(usize, Option<usize>)]) -> Result<Self, BoundsError> {
    let mut ordered = BTreeMap::new();
    for target in targets {
      if !target.tolerance.is_finite() || target.tolerance < 0.0 {
        return Err(BoundsError::InvalidGradientTolerance(target.tolerance));
      }
      let index = differentiated
        .binary_search_by_key(&target.input, |&(input, _)| input)
        .map_err(|_| BoundsError::UnknownGradientInput(target.input))?;
      if ordered.insert(index, target.tolerance).is_some() {
        return Err(BoundsError::DuplicateGradientTarget(target.input));
      }
    }
    let selected = ordered.keys().map(|&index| differentiated[index]).collect();
    let tracked = ordered
      .into_iter()
      .map(|(index, tolerance)| {
        (
          index,
          tolerance,
          ProbabilityInterval {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
          },
        )
      })
      .collect();
    Ok(Self { selected, tracked })
  }

  fn observe(&mut self, bounds: &[GradientBound]) -> bool {
    let mut ready = true;
    for ((_, tolerance, interval), bound) in self.tracked.iter_mut().zip(bounds) {
      *interval = interval.intersect(ProbabilityInterval {
        lower: bound.lower,
        upper: bound.upper,
      });
      let width = interval.upper - interval.lower;
      ready &=
        interval.lower.is_finite() && interval.upper.is_finite() && (width == 0.0 || width.next_up() <= *tolerance);
    }
    ready
  }

  fn merge(&self, gradients: &mut [GradientBound]) {
    for &(index, _, interval) in &self.tracked {
      let gradient = &mut gradients[index];
      gradient.lower = gradient.lower.max(interval.lower);
      gradient.upper = gradient.upper.min(interval.upper);
    }
  }
}

impl Default for GroundRefinement {
  fn default() -> Self {
    Self {
      activation: None,
      lookahead: None,
      active: None,
      next: 0,
      interval: ProbabilityInterval { lower: 0.0, upper: 1.0 },
    }
  }
}

impl GroundRefinement {
  fn stop(active: usize, cap: usize, total: usize, converged: bool) -> Option<BoundsStop> {
    if active == total {
      Some(BoundsStop::Complete)
    } else if converged {
      Some(BoundsStop::Tolerance)
    } else if active == cap {
      Some(BoundsStop::FactLimit)
    } else {
      None
    }
  }

  fn certificate(&self, active_facts: usize, stop_reason: BoundsStop) -> BoundsResult {
    BoundsResult {
      interval: self.interval,
      active_facts,
      stop_reason,
      gradients: Vec::new(),
      exact_probability: None,
      hessian: None,
    }
  }

  fn advance(
    &mut self,
    prepared: &PreparedRelation,
    relation: &str,
    query: &Tuple,
    options: BoundsOptions,
    expired: Option<&dyn Fn() -> bool>,
    goal: GroundGoal<'_>,
  ) -> Result<BoundsResult, BoundsError> {
    let BoundsOptions {
      tolerance,
      max_facts,
      fact_order,
      checkpoint_schedule,
      ..
    } = options;
    let PreparedRelation {
      program,
      positive,
      facts,
      model,
      differentiated,
      choices,
    } = prepared;
    let cap = model.costs.partition_point(|&cost| cost <= max_facts) - 1;
    let total = model.costs.len() - 1;
    let mut convergence = goal
      .gradients
      .map(|targets| GradientConvergence::new(targets, differentiated))
      .transpose()?;
    let refining = tolerance.is_some() || convergence.is_some();
    let probability_only = tolerance.is_some() && convergence.is_none();
    if let Some(active) = self.active
      && let Some(stop_reason) = Self::stop(active, cap, total, tolerance.is_some_and(|t| self.interval.within(t)))
    {
      return Ok(self.certificate(model.costs[active], stop_reason));
    }
    check_interruption(expired)?;
    if !refining {
      self.next = cap;
    }
    let mut initial = None;
    let can_stop_early = tolerance.is_some_and(|t| t < 1.0) || convergence.is_some();
    let partial_gradient_budget = !refining && cap < model.weights.len();
    if self.activation.is_none()
      && fact_order != FactOrder::Input
      && *positive
      && model.groups.is_empty()
      && cap > 0
      && model.weights.len() > 1
      && (can_stop_early || partial_gradient_budget)
    {
      let required = if matches!(fact_order, FactOrder::Required | FactOrder::Lookahead) {
        let tag = evaluate_query(
          program,
          facts,
          relation,
          query,
          RequiredProvenance,
          expired,
          FactTag::Variable,
        )?;
        initial = Some(tag.initial_bounds());
        tag.0.unwrap_or_default()
      } else {
        Vec::new()
      };
      let (ranked, order) = model.rank(&required);
      self.activation = Some(ranked);
      let probe_at = if refining { 4 } else { cap };
      if fact_order == FactOrder::Lookahead && probe_at >= 4 && probe_at < order.len() && probe_at > required.len() {
        self.lookahead = Some((probe_at, order));
      }
    }
    let checkpoint = |activation: &[usize], active| {
      with_choice_checkpoint(choices, active, |provenance| {
        evaluate_query(program, facts, relation, query, provenance, expired, |id| {
          if activation[id] <= active {
            FactTag::Variable(id)
          } else {
            FactTag::Unknown
          }
        })
      })
    };
    let counter = CategoricalModel::new(
      &model.groups,
      |id: &usize| ProbabilityInterval::point(model.weights[*id]),
      interval_weight,
    );
    let adaptive = checkpoint_schedule == CheckpointSchedule::Adaptive
      && *positive
      && model.groups.is_empty()
      && fact_order != FactOrder::Lookahead;
    loop {
      check_interruption(expired)?;
      let active = self.next.min(cap);
      let mut formula = if active == 0
        && let Some(bounds) = initial.take()
      {
        bounds
      } else {
        checkpoint(self.activation.as_deref().unwrap_or(&model.activation), active)?
      };
      let previous = self.interval;
      let mut interval = previous;
      let probe = self.lookahead.as_ref().is_some_and(|(probe_at, _)| *probe_at == active);
      check_interruption(expired)?;
      if probability_only || probe {
        interval = previous.intersect(formula.count(&counter));
      }
      check_interruption(expired)?;
      if probe && !interval.within(tolerance.unwrap_or(0.0)) {
        let (_, order) = self.lookahead.as_ref().expect("lookahead at this checkpoint");
        let baseline = order[active - 1];
        let mut best = baseline;
        let activation = self
          .activation
          .as_mut()
          .expect("lookahead has a ranked activation order");
        // Compare equal-size prefixes. Only the newest slot is swapped, so
        // all facts chosen at earlier checkpoints stay active.
        for &candidate in order.iter().skip(active).take(2) {
          activation.swap(baseline, candidate);
          let alternative = checkpoint(activation, active);
          activation.swap(baseline, candidate);
          let alternative = alternative?;
          check_interruption(expired)?;
          let candidate_interval = previous.intersect(alternative.count(&counter));
          check_interruption(expired)?;
          if candidate_interval.upper - candidate_interval.lower < interval.upper - interval.lower {
            formula = alternative;
            interval = candidate_interval;
            best = candidate;
          }
          if interval.within(tolerance.unwrap_or(0.0)) {
            break;
          }
        }
        activation.swap(baseline, best);
      }
      let mut gradients_ready = true;
      let checked = convergence.as_mut().map(|convergence| {
        let checked = gradient_bounds(&formula, model, &convergence.selected);
        interval = interval.intersect(checked.0);
        gradients_ready = convergence.observe(&checked.1);
        checked
      });
      self.active = Some(active);
      self.interval = interval;
      if active == self.next {
        let factor = if adaptive && interval == previous { 4 } else { 2 };
        self.next = active.saturating_mul(factor).max(1);
      }
      let converged = if probability_only {
        tolerance.is_some_and(|t| interval.within(t))
      } else {
        convergence.is_some() && gradients_ready && tolerance.is_none_or(|t| interval.within(t))
      };
      if let Some(stop_reason) = Self::stop(active, cap, total, converged) {
        let mut result = self.certificate(model.costs[active], stop_reason);
        if !probability_only {
          let (final_interval, gradients, exact) = checked
            .filter(|(_, partials, _)| partials.len() == differentiated.len())
            .unwrap_or_else(|| gradient_bounds(&formula, model, differentiated));
          result.interval = result.interval.intersect(final_interval);
          result.gradients = gradients;
          result.exact_probability = exact;
          if let Some(convergence) = &convergence {
            convergence.merge(&mut result.gradients);
          }
          if goal.hessian && result.exact_probability.is_some() {
            result.hessian = Some(resolved_derivatives(
              &formula.0.as_boolean_formula(),
              model,
              differentiated,
              &mut result.gradients,
            )?);
          }
        }
        return Ok(result);
      }
    }
  }
}

/// Bound a relation with a shared symbolic prefix. With a probability tolerance,
/// return rows with a nonzero upper probability. Without a tolerance, compute
/// gradients at the fact cap and also retain rows with possible sensitivity.
pub fn evaluate_relation(
  program: &Program,
  inputs: impl IntoIterator<Item = BoundsInput>,
  relation: &str,
  options: BoundsOptions,
) -> Result<Vec<(Tuple, BoundsResult)>, BoundsError> {
  if options
    .tolerance
    .is_some_and(|tolerance| !(0.0..=1.0).contains(&tolerance))
  {
    return Err(BoundsError::InvalidTolerance);
  }
  if matches!(options.fact_order, FactOrder::Required | FactOrder::Lookahead) {
    return Err(BoundsError::Unsupported(
      "query-specific ordering for whole-relation bounds; use input or weight",
    ));
  }
  let PreparedRelation {
    program,
    positive,
    facts,
    mut model,
    differentiated,
    choices,
  } = prepare_relation(program, inputs, relation, options.normalize_disjunctions)?;
  let cap = model.costs.partition_point(|&cost| cost <= options.max_facts) - 1;
  let can_rank = positive && model.groups.is_empty();
  if can_rank && options.fact_order == FactOrder::Weight && cap > 0 && options.tolerance.is_none_or(|t| t < 1.0) {
    (model.activation, _) = model.rank(&[]);
  }
  let adaptive = can_rank && options.checkpoint_schedule == CheckpointSchedule::Adaptive;
  let checkpoint = |active| {
    with_choice_checkpoint(&choices, active, |provenance| {
      execute_bounds(&program, &facts, provenance, None, |id| {
        if model.activation[id] <= active {
          FactTag::Variable(id)
        } else {
          FactTag::Unknown
        }
      })
    })
  };
  let Some(tolerance) = options.tolerance else {
    let execution = checkpoint(cap)?;
    let stop_reason = if cap == model.costs.len() - 1 {
      BoundsStop::Complete
    } else {
      BoundsStop::FactLimit
    };
    return Ok(
      execution
        .idb
        .get_internal_collection(relation)
        .into_iter()
        .flat_map(|collection| collection.iter())
        .filter_map(|row| {
          let (interval, gradients, exact_probability) = gradient_bounds(&row.tag, &model, &differentiated);
          (interval.upper > 0.0 || gradients.iter().any(|grad| grad.lower < 0.0 || grad.upper > 0.0)).then_some((
            row.tuple,
            BoundsResult {
              interval,
              active_facts: model.costs[cap],
              stop_reason,
              gradients,
              exact_probability,
              hessian: None,
            },
          ))
        })
        .collect(),
    );
  };
  let counter = model.counter(|id| ProbabilityInterval::point(model.weights[*id]), interval_weight);
  let mut active = 0;
  let mut previous = BTreeMap::<Tuple, ProbabilityInterval>::new();
  loop {
    let execution = checkpoint(active)?;
    let mut rows = BTreeMap::new();
    if let Some(collection) = execution.idb.get_internal_collection(relation) {
      for row in collection.iter() {
        let mut interval = row.tag.count(&counter);
        if let Some(earlier) = previous.get(&row.tuple) {
          interval = earlier.intersect(interval);
        }
        if interval.upper > 0.0 {
          rows.insert(row.tuple, interval);
        }
      }
    }
    let stop = if active == model.costs.len() - 1 {
      Some(BoundsStop::Complete)
    } else if rows.values().all(|interval| interval.within(tolerance)) {
      Some(BoundsStop::Tolerance)
    } else if active == cap {
      Some(BoundsStop::FactLimit)
    } else {
      None
    };
    if let Some(stop_reason) = stop {
      return Ok(
        rows
          .into_iter()
          .map(|(tuple, interval)| {
            (
              tuple,
              BoundsResult {
                interval,
                active_facts: model.costs[active],
                stop_reason,
                gradients: Vec::new(),
                exact_probability: None,
                hessian: None,
              },
            )
          })
          .collect(),
      );
    }
    let factor = if adaptive && rows == previous { 4 } else { 2 };
    previous = rows;
    active = active.saturating_mul(factor).max(1).min(cap);
  }
}

struct PreparedRelation {
  program: Program,
  positive: bool,
  facts: Facts,
  model: BoundsModel,
  differentiated: Vec<(usize, Option<usize>)>,
  choices: Option<Arc<ChoiceTable>>,
}

fn prepare_relation(
  program: &Program,
  inputs: impl IntoIterator<Item = BoundsInput>,
  relation: &str,
  normalize_disjunctions: bool,
) -> Result<PreparedRelation, BoundsError> {
  let mut program = query_program(program, relation)?;
  let (positive, has_choices) = validate_program(&program)?;
  let file_inputs = if program.relations().any(|relation| relation.input_file.is_some()) {
    let mut registry: InputFileRegistry = InputFileRegistry::new();
    registry
      .load_files(program.relations().filter_map(|relation| relation.input_file.as_ref()))
      .map_err(DatabaseError::IO)?;
    Some((registry, RuntimeEnvironment::default()))
  } else {
    None
  };
  let mut model = ModelBuilder::default();
  for rel in program
    .strata
    .iter_mut()
    .flat_map(|stratum| stratum.relations.values_mut())
  {
    if let Some(file) = rel.input_file.take() {
      let (registry, env) = file_inputs.as_ref().expect("file inputs are loaded before parsing");
      let content = registry.get(file.file_path()).expect("requested file is loaded");
      let facts = parse_input_file_facts(env, &file, content, &rel.tuple_type).map_err(DatabaseError::IO)?;
      rel
        .facts
        .extend(facts.into_iter().map(|(tag, tuple)| Fact { tag, tuple }));
    }
    for fact in std::mem::take(&mut rel.facts) {
      let (probability, group) = match fact.tag {
        DynamicInputTag::None => (1.0, None),
        DynamicInputTag::Bool(value) => (if value { 1.0 } else { 0.0 }, None),
        DynamicInputTag::Float(p) => (p, None),
        DynamicInputTag::Exclusive(id) => (1.0, Some(GroupKey::Program(id))),
        DynamicInputTag::ExclusiveFloat(p, id) | DynamicInputTag::ExclusiveFloatWithID(_, p, id) => {
          (p, Some(GroupKey::Program(id)))
        }
        _ => {
          return Err(BoundsError::Unsupported(
            "input tags other than probabilistic facts and choices",
          ))
        }
      };
      model.push(Some((rel.predicate.clone(), fact.tuple)), probability, group, false)?;
    }
  }
  let inputs: Vec<_> = inputs.into_iter().collect();
  let visible = |rows: &BoundsRows| rows.as_slice().iter().any(|(name, _)| program.relation(name).is_some());
  let relevant_groups: HashSet<_> = inputs
    .iter()
    .filter(|fact| visible(&fact.rows))
    .filter_map(|fact| fact.group)
    .collect();
  let mut differentiated = Vec::new();
  for (input, fact) in inputs.into_iter().enumerate() {
    let variable = if visible(&fact.rows) || fact.group.is_some_and(|id| relevant_groups.contains(&id)) {
      model.push(
        fact
          .rows
          .into_rows()
          .filter(|(name, _)| program.relation(name).is_some()),
        fact.probability,
        fact.group.map(GroupKey::Input),
        fact.differentiate,
      )?
    } else {
      None
    };
    if fact.differentiate {
      differentiated.push((input, variable));
    }
  }
  let (facts, mut model) = model.finish(normalize_disjunctions)?;
  let choices = if has_choices {
    // Unknown choices and uncertain inputs overapproximate all reachable
    // groundings, including when the body uses stratified negation.
    let support = Rc::new(RefCell::new(ChoiceSupport::new()));
    execute_bounds(
      &program,
      &facts,
      BoundsProvenance {
        choices: Some(ChoiceState::Discover(Rc::clone(&support))),
      },
      None,
      |_| FactTag::Unknown,
    )?;
    Some(Arc::new(model.add_choices(support.take())?))
  } else {
    None
  };
  Ok(PreparedRelation {
    program,
    positive,
    facts,
    model,
    differentiated,
    choices,
  })
}

fn query_program(program: &Program, relation: &str) -> Result<Program, BoundsError> {
  let graph = program.dependency_graph();
  let target = graph
    .node_indices()
    .find(|&node| graph[node].predicate == relation)
    .ok_or_else(|| BoundsError::UnknownRelation(relation.into()))?;
  let mut walk = Dfs::new(Reversed(&graph), target);
  let keep: HashSet<_> = std::iter::from_fn(|| walk.next(Reversed(&graph)))
    .map(|node| &graph[node].predicate)
    .collect();

  let mut query = program.clone();
  query.relation_to_stratum.retain(|name, _| keep.contains(name));
  for stratum in &mut query.strata {
    stratum.relations.retain(|name, _| keep.contains(name));
    stratum.updates.retain(|update| keep.contains(&update.target));
  }
  query.set_output_relations(vec![relation]);
  Ok(query)
}

type Facts = BTreeMap<String, Vec<(Option<usize>, Tuple)>>;
type ChoiceKey = (usize, Tuple);
type ChoiceSupport = BTreeMap<ChoiceKey, BTreeSet<Tuple>>;
type ChoiceTable = BTreeMap<ChoiceKey, BTreeMap<Tuple, ChoiceOutcome>>;

#[derive(Clone, Copy)]
enum ChoiceOutcome {
  Certain,
  Symbolic { id: usize, activation: usize },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum GroupKey {
  Program(usize),
  Input(BoundsInputGroup),
}

#[derive(Default)]
struct ModelBuilder {
  facts: Facts,
  weights: Vec<f64>,
  groups: Vec<Vec<(usize, f64)>>,
  group_indices: BTreeMap<GroupKey, (usize, usize)>,
  units: Vec<Option<usize>>,
  activation: Vec<usize>,
  fact_groups: Vec<Option<usize>>,
}

impl ModelBuilder {
  fn push(
    &mut self,
    rows: impl IntoIterator<Item = (String, Tuple)>,
    probability: f64,
    group: Option<GroupKey>,
    differentiate: bool,
  ) -> Result<Option<usize>, BoundsError> {
    if !(0.0..=1.0).contains(&probability) {
      return Err(BoundsError::InvalidProbability(probability));
    }
    if group.is_none() && !differentiate {
      if probability == 0.0 {
        return Ok(None);
      }
      if probability == 1.0 {
        for (relation, tuple) in rows {
          self.facts.entry(relation).or_default().push((None, tuple));
        }
        return Ok(None);
      }
    }
    let id = self.weights.len();
    let (group, activation) = if let Some(key) = group {
      let (group, activation) = *self.group_indices.entry(key).or_insert_with(|| {
        let group = self.groups.len();
        self.groups.push(Vec::new());
        self.units.push(Some(group));
        (group, self.units.len())
      });
      self.groups[group].push((id, probability));
      (Some(group), activation)
    } else {
      self.units.push(None);
      (None, self.units.len())
    };
    self.activation.push(activation);
    self.weights.push(probability);
    self.fact_groups.push(group);
    for (relation, tuple) in rows {
      self.facts.entry(relation).or_default().push((Some(id), tuple));
    }
    Ok(Some(id))
  }

  fn finish(self, normalize: bool) -> Result<(Facts, BoundsModel), BoundsError> {
    let mut costs = vec![0];
    for group in self.units {
      costs.push(costs.last().unwrap() + group.map_or(1, |group| self.groups[group].len()));
    }
    let groups = self
      .groups
      .into_iter()
      .map(|facts| CategoricalGroup::new(facts, normalize))
      .collect::<Result<Vec<_>, _>>()?;
    Ok((
      self.facts,
      BoundsModel {
        weights: self.weights,
        groups,
        activation: self.activation,
        fact_groups: self.fact_groups,
        costs,
        normalize,
      },
    ))
  }
}

struct BoundsModel {
  weights: Vec<f64>,
  groups: Vec<CategoricalGroup>,
  activation: Vec<usize>,
  fact_groups: Vec<Option<usize>>,
  costs: Vec<usize>,
  normalize: bool,
}

impl BoundsModel {
  fn add_choices(&mut self, support: ChoiceSupport) -> Result<ChoiceTable, BoundsError> {
    let mut choices = BTreeMap::new();
    for (key, alternatives) in support {
      let count = alternatives.len();
      let outcomes = if count == 1 {
        alternatives
          .into_iter()
          .map(|alt| (alt, ChoiceOutcome::Certain))
          .collect()
      } else {
        let activation = self.costs.len();
        let group = self.groups.len();
        let ids = self.weights.len()..self.weights.len() + count;
        // Normalizing equal unit weights gives exact uniform mass, even for 1/3.
        self
          .groups
          .push(CategoricalGroup::new(ids.clone().map(|id| (id, 1.0)).collect(), true)?);
        self.weights.extend(std::iter::repeat_n(1.0, count));
        self.activation.extend(std::iter::repeat_n(activation, count));
        self.fact_groups.extend(std::iter::repeat_n(Some(group), count));
        self.costs.push(self.costs[activation - 1] + count);
        alternatives
          .into_iter()
          .zip(ids)
          .map(|(alt, id)| (alt, ChoiceOutcome::Symbolic { id, activation }))
          .collect()
      };
      choices.insert(key, outcomes);
    }
    Ok(choices)
  }

  fn rank(&self, required: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<_> = (0..self.weights.len()).collect();
    order.sort_unstable_by(|&a, &b| {
      required
        .binary_search(&b)
        .is_ok()
        .cmp(&required.binary_search(&a).is_ok())
        .then_with(|| self.weights[b].total_cmp(&self.weights[a]))
        .then_with(|| a.cmp(&b))
    });
    let mut activation = vec![0; order.len()];
    for (position, &id) in order.iter().enumerate() {
      activation[id] = position + 1;
    }
    (activation, order)
  }

  fn counter<E, W, T>(&self, independent: W, selector: T) -> CategoricalModel<'_, W, T>
  where
    W: Fn(&usize) -> E,
    T: Fn(ProbabilityWeight) -> E,
  {
    CategoricalModel::new(&self.groups, independent, selector)
  }
}

fn interval_weight(weight: ProbabilityWeight) -> ProbabilityInterval {
  ProbabilityInterval {
    lower: weight.lower,
    upper: weight.upper,
  }
}

fn evaluate_query<P: Provenance<InputTag = FactTag>>(
  program: &Program,
  facts: &Facts,
  relation: &str,
  query: &Tuple,
  provenance: P,
  interrupted: Option<&dyn Fn() -> bool>,
  tag_input: impl Fn(usize) -> FactTag,
) -> Result<P::Tag, BoundsError> {
  let zero = provenance.zero();
  let execution = execute_bounds(program, facts, provenance, interrupted, tag_input)?;
  Ok(
    execution
      .idb
      .get_internal_collection(relation)
      .and_then(|rows| rows.iter().find(|row| row.tuple == *query).map(|row| row.tag))
      .unwrap_or(zero),
  )
}

fn execute_bounds<P: Provenance<InputTag = FactTag>>(
  program: &Program,
  facts: &Facts,
  mut provenance: P,
  interrupted: Option<&dyn Fn() -> bool>,
  tag_input: impl Fn(usize) -> FactTag,
) -> Result<DynamicExecutionContext<P>, BoundsError> {
  let mut execution =
    DynamicExecutionContext::<P>::new_with_program_and_options(program.clone(), ExecutionOptions::default());
  for rel in program.relations().filter(|rel| rel.immutable) {
    execution.edb.tap_relation(&rel.predicate);
  }
  // IDs are assigned once, before refinement. Repeated occurrences of a fact
  // in different proofs must refer to the same Boolean variable.
  for (name, facts) in facts {
    check_interruption(interrupted)?;
    let rows = facts
      .iter()
      .map(|(variable, tuple)| (variable.map(&tag_input), tuple.clone()))
      .collect();
    execution.edb.add_static_input_facts(name, rows)?;
  }
  let runtime = RuntimeEnvironment::new(
    program.function_registry.clone(),
    program.predicate_registry.clone(),
    program.aggregate_registry.clone(),
  );
  if let Some(interrupted) = interrupted {
    execution.execute_interruptible(&runtime, &provenance, interrupted)?;
  } else {
    execution.execute(&runtime, &mut provenance)?;
  }
  Ok(execution)
}

fn gradient_bounds(
  formula: &BoundsTag,
  model: &BoundsModel,
  differentiated: &[(usize, Option<usize>)],
) -> (ProbabilityInterval, Vec<GradientBound>, Option<f64>) {
  let event = formula.0.as_boolean_formula();
  if model.groups.is_empty() {
    let upper = (formula.0 != formula.1).then(|| formula.1.as_boolean_formula());
    with_prepared_events(&event, upper.as_ref(), |lower, upper| {
      gradient_bounds_for_events(formula, &event, model, differentiated, Some((lower, upper)))
    })
  } else {
    gradient_bounds_for_events(formula, &event, model, differentiated, None)
  }
}

fn gradient_bounds_for_events(
  formula: &BoundsTag,
  event: &BooleanFormula,
  model: &BoundsModel,
  differentiated: &[(usize, Option<usize>)],
  prepared: Option<(&PreparedWmc<'_>, Option<&PreparedWmc<'_>>)>,
) -> (ProbabilityInterval, Vec<GradientBound>, Option<f64>) {
  let ids: HashSet<_> = formula
    .0
    .iter()
    .chain(formula.1.iter())
    .flat_map(Clause::iter)
    .map(|literal| literal.fact_id())
    .collect();
  let used_groups: HashSet<_> = ids.iter().filter_map(|&id| model.fact_groups[id]).collect();
  let rounded = |id: &usize| ProbabilityInterval::point(model.weights[*id]);
  let real = |id: &usize| model.weights[*id];
  type Cofactors = (ProbabilityInterval, HashMap<usize, [ProbabilityInterval; 2]>);
  let batch: Option<(Cofactors, Option<Cofactors>)> = {
    #[cfg(not(target_family = "wasm"))]
    {
      prepared
        .filter(|_| {
          differentiated
            .iter()
            .filter_map(|&(_, id)| id)
            .filter(|id| ids.contains(id))
            .take(8)
            .count()
            == 8
        })
        .map(|(lower, upper)| {
          (
            lower.cofactors(&Rounded, &rounded),
            upper.map(|event| event.cofactors(&Rounded, &rounded)),
          )
        })
    }
    #[cfg(target_family = "wasm")]
    {
      None
    }
  };
  let certified = |forced: Option<(usize, f64)>| {
    if let Some((lower, upper)) = &batch
      && let Some((id, value)) = forced
    {
      let branch = usize::from(value == 1.0);
      let select = |(mass, cofactors): &Cofactors| cofactors.get(&id).map_or(*mass, |pair| pair[branch]);
      return ProbabilityInterval {
        lower: select(lower).lower,
        upper: select(upper.as_ref().unwrap_or(lower)).upper,
      };
    }
    let weight = |id: &usize| match forced {
      Some((fact, value)) if fact == *id => ProbabilityInterval::point(value),
      _ => rounded(id),
    };
    if let Some((lower, upper)) = prepared {
      let lower = lower.count(&Rounded, &weight);
      let upper = upper.map_or(lower, |event| event.count(&Rounded, &weight));
      ProbabilityInterval {
        lower: lower.lower,
        upper: upper.upper,
      }
    } else {
      formula.count(&model.counter(weight, interval_weight))
    }
  };
  let unconditioned = certified(None);
  let exact = (formula.0 == formula.1).then(|| {
    let variables: HashSet<_> = differentiated.iter().filter_map(|&(_, id)| id).collect();
    let algebra = DualNumberSemiring::new(model.weights.len());
    let weight = |id: &usize| {
      if variables.contains(id) {
        algebra.singleton(model.weights[*id], *id)
      } else {
        algebra.constant(model.weights[*id])
      }
    };
    if let Some((lower, _)) = prepared {
      lower.count(&algebra, &weight)
    } else {
      let counter = model.counter(weight, |weight: ProbabilityWeight| algebra.constant(weight.nearest));
      count(event, &counter, &algebra, &DefaultWmcBackend)
    }
  });
  let categorical_reference = |group| {
    if model.normalize {
      (unconditioned, exact.as_ref().map(|value| value.real))
    } else {
      (
        formula.count(&model.counter(rounded, interval_weight).force(group, None)),
        exact.as_ref().map(|_| {
          let counter = model
            .counter(real, |weight: ProbabilityWeight| weight.nearest)
            .force(group, None);
          count(event, &counter, &RealSemiring, &DefaultWmcBackend)
        }),
      )
    }
  };
  let references: BTreeMap<_, _> = differentiated
    .iter()
    .filter_map(|&(_, id)| model.fact_groups[id?])
    .filter(|group| used_groups.contains(group))
    .collect::<HashSet<_>>()
    .into_iter()
    .map(|group| (group, categorical_reference(group)))
    .collect();
  let gradients = differentiated
    .iter()
    .map(|&(input, variable)| {
      let ((lower, upper), derivative) =
        if let Some((id, group)) = variable.and_then(|id| model.fact_groups[id].map(|group| (id, group))) {
          let (bounds, derivative) = if let Some(&(reference, reference_value)) = references.get(&group) {
            let selected = formula.count(&model.counter(rounded, interval_weight).force(group, Some(id)));
            let derivative = reference_value.map(|reference| {
              let counter = model
                .counter(real, |weight: ProbabilityWeight| weight.nearest)
                .force(group, Some(id));
              count(event, &counter, &RealSemiring, &DefaultWmcBackend) - reference
            });
            (difference(selected, reference), derivative)
          } else if exact.is_none() {
            (difference(unconditioned, unconditioned), None)
          } else {
            ((0.0, 0.0), Some(0.0))
          };
          if model.normalize {
            let total = model.groups[group].total_weight();
            (divide(bounds, total), derivative.map(|value| value / total.nearest))
          } else {
            (bounds, derivative)
          }
        } else {
          let derivative = exact
            .as_ref()
            .map(|value| variable.and_then(|id| value.deriv.get(id)).copied().unwrap_or(0.0));
          let bounds = match variable {
            Some(id) if ids.contains(&id) => {
              // Count cofactors, including at the endpoints of the input's domain.
              difference(certified(Some((id, 1.0))), certified(Some((id, 0.0))))
            }
            // A variable absent from both formulas can still affect the query in
            // the worlds where the two formulas disagree.
            Some(_) if exact.is_none() => difference(unconditioned, unconditioned),
            _ => (0.0, 0.0),
          };
          (bounds, derivative)
        };
      GradientBound {
        input,
        lower,
        upper,
        exact: derivative,
      }
    })
    .collect();
  (unconditioned, gradients, exact.map(|value| value.real))
}

fn divide((lower, upper): (f64, f64), denominator: ProbabilityWeight) -> (f64, f64) {
  (
    if lower == 0.0 {
      0.0
    } else {
      (lower
        / if lower > 0.0 {
          denominator.upper
        } else {
          denominator.lower
        })
      .next_down()
    },
    if upper == 0.0 {
      0.0
    } else {
      (upper
        / if upper > 0.0 {
          denominator.lower
        } else {
          denominator.upper
        })
      .next_up()
    },
  )
}

#[derive(Clone, Copy)]
enum ForcedOutcome {
  Unconditioned,
  Bernoulli(usize, bool),
  Categorical(usize, Option<usize>),
}

impl ForcedOutcome {
  fn restrict(self, event: &BooleanFormula, model: &BoundsModel) -> BooleanFormula {
    if matches!(self, Self::Unconditioned) {
      return event.clone();
    }
    use BooleanFormula::*;
    match event {
      Literal(literal) => {
        let id = literal.fact_id();
        let selected = match self {
          Self::Bernoulli(fact, value) if id == fact => Some(value),
          Self::Categorical(group, selected) if model.fact_groups[id] == Some(group) => Some(Some(id) == selected),
          _ => None,
        };
        selected.map_or_else(|| Literal(literal.clone()), |value| Constant(value == literal.sign()))
      }
      Not(term) => Not(Box::new(self.restrict(term, model))),
      And(terms) => BooleanFormula::conjunction(terms.iter().map(|term| self.restrict(term, model))),
      Or(terms) => BooleanFormula::disjunction(terms.iter().map(|term| self.restrict(term, model))),
      Constant(value) => Constant(*value),
    }
  }
}

fn resolved_derivatives(
  event: &BooleanFormula,
  model: &BoundsModel,
  differentiated: &[(usize, Option<usize>)],
  gradients: &mut [GradientBound],
) -> Result<Vec<(usize, usize, f64)>, BoundsError> {
  let ids: HashSet<_> = event.fact_ids().into_iter().collect();
  let groups: HashSet<_> = ids.iter().filter_map(|&id| model.fact_groups[id]).collect();
  let relevant: Vec<_> = differentiated
    .iter()
    .enumerate()
    .filter_map(|(index, &(_, id))| {
      let id = id?;
      let group = model.fact_groups[id];
      let relevant = group
        .filter(|_| model.normalize)
        .map_or_else(|| ids.contains(&id), |group| groups.contains(&group));
      relevant.then_some((index, id, group))
    })
    .collect();
  if relevant.is_empty() {
    return Ok(Vec::new());
  }

  use ForcedOutcome::{Bernoulli, Categorical, Unconditioned};
  let contrast = |id: usize, group: Option<usize>| match group {
    Some(group) if model.normalize => (
      Categorical(group, Some(id)),
      Unconditioned,
      model.groups[group].total_weight().nearest,
    ),
    Some(group) => (Categorical(group, Some(id)), Categorical(group, None), 1.0),
    None => (Bernoulli(id, true), Bernoulli(id, false), 1.0),
  };
  let counter = model.counter(|id| model.weights[*id], |weight: ProbabilityWeight| weight.nearest);
  let mass = |event| count(&event, &counter, &RealSemiring, &DefaultWmcBackend);
  let and = |a, b| BooleanFormula::conjunction([a, b]);
  let or = |a, b| BooleanFormula::disjunction([a, b]);
  let not = |a| BooleanFormula::Not(Box::new(a));
  let signed = |positive: BooleanFormula, negative: BooleanFormula| {
    mass(and(positive.clone(), not(negative.clone()))) - mass(and(negative, not(positive)))
  };
  // Expected positive part of a difference between two sums of indicators.
  let exceeds = |p1: &BooleanFormula, p2: &BooleanFormula, n1: &BooleanFormula, n2: &BooleanFormula| {
    mass(or(and(p1.clone(), not(n1.clone())), and(p2.clone(), not(n2.clone()))))
      + mass(and(p2.clone(), not(n1.clone())))
  };

  let mut contrasts = HashMap::new();
  let mut entries = Vec::new();
  for (offset, &(i, id, group)) in relevant.iter().enumerate() {
    let (a, a_ref, a_scale) = contrast(id, group);
    let yes = a.restrict(event, model);
    let no = a_ref.restrict(event, model);
    let first = *contrasts.entry(id).or_insert_with(|| signed(yes.clone(), no.clone()));
    let derivative = first / a_scale;
    if !derivative.is_finite() {
      return Err(BoundsError::NonFiniteDerivative);
    }
    gradients[i].exact = Some(derivative);
    for &(j, other, other_group) in &relevant[offset..] {
      let entry = if let Some(group) = group.filter(|_| group == other_group) {
        if model.normalize {
          // Within a normalized group, d²P/dw_i dw_j = -(dP/dw_i + dP/dw_j) / total.
          let right = *contrasts
            .entry(other)
            .or_insert_with(|| signed(Categorical(group, Some(other)).restrict(event, model), event.clone()));
          -(first + right) / a_scale / a_scale
        } else {
          0.0
        }
      } else if id == other {
        0.0
      } else {
        let (b, b_ref, b_scale) = contrast(other, other_group);
        let (pp, nn) = (b.restrict(&yes, model), b_ref.restrict(&no, model));
        let (pn, np) = (b_ref.restrict(&yes, model), b.restrict(&no, model));
        let (p1, p2) = (or(pp.clone(), nn.clone()), and(pp, nn));
        let (n1, n2) = (or(pn.clone(), np.clone()), and(pn, np));
        // Cancel pointwise before counting, so common events cannot hide rare effects.
        let term = exceeds(&p1, &p2, &n1, &n2) - exceeds(&n1, &n2, &p1, &p2);
        term / a_scale.max(b_scale) / a_scale.min(b_scale)
      };
      if !entry.is_finite() {
        return Err(BoundsError::NonFiniteDerivative);
      }
      if entry != 0.0 {
        entries.push((i, j, entry));
      }
    }
  }
  Ok(entries)
}

fn difference(positive: ProbabilityInterval, negative: ProbabilityInterval) -> (f64, f64) {
  (
    (positive.lower - negative.upper).next_down().max(-1.0),
    (positive.upper - negative.lower).next_up().min(1.0),
  )
}

fn validate_program(program: &Program) -> Result<(bool, bool), BoundsError> {
  fn supported_type(ty: &TupleType) -> bool {
    match ty {
      TupleType::Tuple(fields) => fields.iter().all(supported_type),
      TupleType::Value(v) => !matches!(v, ValueType::Entity | ValueType::Tensor | ValueType::Symbol),
    }
  }

  for relation in program.relations() {
    if !matches!(relation.scheduler, None | Some(Scheduler::LFP)) {
      return Err(BoundsError::Unsupported("non-LFP scheduler attributes"));
    }
    if relation.storage != StorageMetadata::Sorted {
      return Err(BoundsError::Unsupported("non-sorted storage"));
    }
    if !supported_type(&relation.tuple_type) {
      return Err(BoundsError::Unsupported("entity, symbol, or tensor values"));
    }
  }
  // RAM expressions do not retain their operand types. Conservatively rule
  // out temporal arithmetic, whose runtime operations can overflow.
  let allow_add_sub = !program.relations().any(|relation| temporal_type(&relation.tuple_type))
    && !program
      .strata
      .iter()
      .flat_map(|s| &s.updates)
      .any(|u| has_temporal_literal(&u.dataflow));
  let mut validator = BoundsValidator {
    program,
    allow_add_sub,
    positive: true,
    has_choices: false,
  };
  for stratum in &program.strata {
    for update in &stratum.updates {
      validator.dataflow(&update.dataflow, stratum.is_recursive)?;
    }
  }
  Ok((validator.positive, validator.has_choices))
}

fn temporal_type(ty: &TupleType) -> bool {
  match ty {
    TupleType::Tuple(fields) => fields.iter().any(temporal_type),
    TupleType::Value(v) => matches!(v, ValueType::DateTime | ValueType::Duration),
  }
}

fn has_temporal_literal(dataflow: &Dataflow) -> bool {
  fn in_expr(expr: &Expr) -> bool {
    match expr {
      Expr::Constant(v) => temporal_type(&TupleType::Value(v.value_type())),
      Expr::Tuple(fields) => fields.iter().any(in_expr),
      Expr::Binary(b) => in_expr(&b.op1) || in_expr(&b.op2),
      Expr::Unary(u) => in_expr(&u.op1),
      Expr::IfThenElse(i) => in_expr(&i.cond) || in_expr(&i.then_br) || in_expr(&i.else_br),
      Expr::Call(c) => c.args.iter().any(in_expr),
      _ => false,
    }
  }
  (match dataflow {
    Dataflow::Unit(ty) => temporal_type(ty),
    Dataflow::UntaggedVec(rows) => rows.iter().any(|row| temporal_type(&row.tuple_type())),
    Dataflow::Project(_, expr) | Dataflow::Filter(_, expr) => in_expr(expr),
    _ => false,
  }) || dataflow.children().any(has_temporal_literal)
}

struct BoundsValidator<'a> {
  program: &'a Program,
  allow_add_sub: bool,
  positive: bool,
  has_choices: bool,
}

impl BoundsValidator<'_> {
  fn expr(&self, expr: &Expr, allow_computed: bool) -> Result<(), BoundsError> {
    let validate = |expr| self.expr(expr, allow_computed);
    match expr {
      Expr::Access(_) | Expr::Constant(_) => Ok(()),
      Expr::Tuple(fields) => fields.iter().try_for_each(validate),
      Expr::Call(c) if allow_computed => {
        let name = c.function.as_str();
        // Besides being pure, a call must respect tuple equality (e.g. signed
        // zeros compare equal here, but atan2 can distinguish them).
        if !self.program.function_registry.is_builtin(name)
          || !matches!(
            name,
            "floor"
              | "ceil"
              | "exp"
              | "exp2"
              | "log"
              | "log2"
              | "sin"
              | "cos"
              | "tan"
              | "asin"
              | "acos"
              | "atan"
              | "string_concat"
              | "string_length"
              | "string_char_at"
              | "string_index_of"
              | "string_upper"
              | "string_lower"
              | "string_replace"
              | "string_trim"
              | "datetime_day"
              | "datetime_month"
              | "datetime_month0"
              | "datetime_year"
          )
        {
          return Err(BoundsError::UnsupportedForeignCall {
            kind: "function",
            name: name.into(),
            reason: "not a supported deterministic built-in",
          });
        }
        c.args.iter().try_for_each(validate)
      }
      Expr::New(_) => Err(BoundsError::Unsupported("entity constructors")),
      _ if !allow_computed => Err(BoundsError::Unsupported("computed projections in recursive strata")),
      Expr::Binary(b) if !b.op.is_arith() || (self.allow_add_sub && matches!(b.op, BinaryOp::Add | BinaryOp::Sub)) => {
        validate(&b.op1)?;
        validate(&b.op2)
      }
      Expr::Unary(u) if matches!(u.op, UnaryOp::Pos | UnaryOp::Not) => validate(&u.op1),
      Expr::IfThenElse(i) => {
        validate(&i.cond)?;
        validate(&i.then_br)?;
        validate(&i.else_br)
      }
      _ => Err(BoundsError::Unsupported(
        "arithmetic operator or cast in this expression",
      )),
    }
  }

  fn predicate(&self, name: &str, recursive: bool) -> Result<(), BoundsError> {
    let registry = &self.program.predicate_registry;
    let error = |reason| BoundsError::UnsupportedForeignCall {
      kind: "predicate",
      name: name.into(),
      reason,
    };
    let predicate = registry
      .get(name)
      .filter(|_| registry.is_builtin(name))
      .ok_or_else(|| error("not a supported deterministic, untagged built-in"))?;
    if !matches!(
      predicate.name().as_str(),
      "range" | "float_eq" | "string_chars" | "string_find" | "string_index" | "string_split" | "datetime_ymd"
    ) {
      return Err(error("not a supported deterministic, untagged built-in"));
    }
    if recursive && predicate.num_free() > 0 {
      return Err(error(
        "value-generating predicates are not supported in recursive strata",
      ));
    }
    Ok(())
  }

  fn predicate_args(&self, args: &[Expr]) -> Result<(), BoundsError> {
    // RAM's bound-call operators read arguments directly, not via expression evaluation.
    if args
      .iter()
      .all(|arg| matches!(arg, Expr::Access(_) | Expr::Constant(_)))
    {
      Ok(())
    } else {
      Err(BoundsError::Unsupported("computed arguments to RAM foreign predicates"))
    }
  }

  fn dataflow(&mut self, dataflow: &Dataflow, recursive: bool) -> Result<(), BoundsError> {
    match dataflow {
      Dataflow::Project(_, expr) => self.expr(expr, !recursive)?,
      Dataflow::Filter(_, expr) => self.expr(expr, true)?,
      Dataflow::Antijoin(..) | Dataflow::Difference(..) => self.positive = false,
      Dataflow::Reduce(reduce) => {
        if reduce.aggregate_info.has_exclamation_mark {
          return Err(BoundsError::Unsupported("snapshot aggregation (`!` variants)"));
        }
        if !self.program.aggregate_registry.is_builtin(&reduce.aggregator)
          || !matches!(
            reduce.aggregator.as_str(),
            "count" | "exists" | "min" | "max" | "argmin" | "argmax" | "sum" | "prod"
          )
        {
          return Err(BoundsError::Unsupported(
            "aggregation outside the supported possible-world reducers",
          ));
        }
        self.positive = false;
      }
      Dataflow::Exclusion(_, alternatives, Some(_)) if matches!(alternatives.as_ref(), Dataflow::UntaggedVec(_)) => {
        self.positive = false;
        self.has_choices = true;
      }
      Dataflow::Exclusion(..) => {
        return Err(BoundsError::Unsupported(
          "rule choices without a shared source head variable pattern",
        ));
      }
      Dataflow::OverwriteOne(_) => return Err(BoundsError::Unsupported("provenance-overwriting rules")),
      Dataflow::JoinIndexedVec(..) => return Err(BoundsError::Unsupported("indexed joins")),
      Dataflow::ForeignPredicateGround(name, _) => self.predicate(name, recursive)?,
      Dataflow::ForeignPredicateConstraint(_, name, args) | Dataflow::ForeignPredicateJoin(_, name, args) => {
        self.predicate(name, recursive)?;
        self.predicate_args(args)?;
      }
      _ => {}
    }
    dataflow
      .children()
      .try_for_each(|child| self.dataflow(child, recursive))
  }
}

#[derive(Clone, Debug)]
enum FactTag {
  Variable(usize),
  Unknown,
}

impl StaticInputTag for FactTag {
  fn from_dynamic_input_tag(_: &DynamicInputTag) -> Option<Self> {
    None
  }
}

// None means no proof; Some(empty) means there are proofs with no fact in common.
// The Boolean records whether any proof uses only deterministic facts.
#[derive(Clone, Debug, PartialEq)]
struct RequiredTag(Option<Vec<usize>>, bool);

impl RequiredTag {
  fn initial_bounds(&self) -> BoundsTag {
    let lower = if self.1 { DNFFormula::one() } else { DNFFormula::zero() };
    let upper = if self.0.is_some() {
      DNFFormula::one()
    } else {
      DNFFormula::zero()
    };
    BoundsTag(lower, upper)
  }
}

impl fmt::Display for RequiredTag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:?}", self.0)
  }
}

impl Tag for RequiredTag {}

#[derive(Clone)]
struct RequiredProvenance;

impl Provenance for RequiredProvenance {
  type InputTag = FactTag;
  type Tag = RequiredTag;
  type OutputTag = RequiredTag;

  fn name(&self) -> String {
    "required_facts".into()
  }

  fn tagging_fn(&self, tag: FactTag) -> RequiredTag {
    match tag {
      FactTag::Variable(id) => RequiredTag(Some(vec![id]), false),
      FactTag::Unknown => unreachable!("required-fact analysis initializes every fact"),
    }
  }

  fn recover_fn(&self, tag: &RequiredTag) -> RequiredTag {
    tag.clone()
  }

  fn discard(&self, tag: &RequiredTag) -> bool {
    tag.0.is_none()
  }

  fn zero(&self) -> RequiredTag {
    RequiredTag(None, false)
  }

  fn one(&self) -> RequiredTag {
    RequiredTag(Some(Vec::new()), true)
  }

  fn add(&self, left: &RequiredTag, right: &RequiredTag) -> RequiredTag {
    match (&left.0, &right.0) {
      (None, _) => right.clone(),
      (_, None) => left.clone(),
      (Some(left_ids), Some(right_ids)) => {
        let mut common = Vec::new();
        if !left_ids.is_empty() && !right_ids.is_empty() {
          let mut right_ids = right_ids.iter().copied().peekable();
          for &id in left_ids {
            while right_ids.next_if(|&other| other < id).is_some() {}
            if right_ids.next_if_eq(&id).is_some() {
              common.push(id);
            }
          }
        }
        RequiredTag(Some(common), left.1 || right.1)
      }
    }
  }

  fn mult(&self, left: &RequiredTag, right: &RequiredTag) -> RequiredTag {
    match (&left.0, &right.0) {
      (Some(left_ids), Some(right_ids)) => {
        let ids = if left_ids.is_empty() {
          right_ids.clone()
        } else if right_ids.is_empty() {
          left_ids.clone()
        } else {
          left_ids.iter().merge(right_ids).dedup().copied().collect()
        };
        RequiredTag(Some(ids), left.1 && right.1)
      }
      _ => self.zero(),
    }
  }

  fn saturated(&self, old: &RequiredTag, new: &RequiredTag) -> bool {
    old == new
  }
}

#[derive(Clone, Debug, PartialEq)]
struct BoundsTag(DNFFormula, DNFFormula);

impl BoundsTag {
  fn count(&self, model: &impl WmcModel<Rounded>) -> ProbabilityInterval {
    let lower = count(&self.0.as_boolean_formula(), model, &Rounded, &DefaultWmcBackend);
    let upper = if self.0 == self.1 {
      lower
    } else {
      count(&self.1.as_boolean_formula(), model, &Rounded, &DefaultWmcBackend)
    };
    ProbabilityInterval {
      lower: lower.lower,
      upper: upper.upper,
    }
  }
}

impl fmt::Display for BoundsTag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}, {}]", self.0, self.1)
  }
}

impl Tag for BoundsTag {}

#[derive(Clone, Default)]
struct BoundsProvenance {
  choices: Option<ChoiceState>,
}

#[derive(Clone)]
enum ChoiceState {
  Discover(Rc<RefCell<ChoiceSupport>>),
  Refine {
    choices: Arc<ChoiceTable>,
    active: usize,
    missing: Rc<Cell<bool>>,
  },
}

fn with_choice_checkpoint<T>(
  choices: &Option<Arc<ChoiceTable>>,
  active: usize,
  run: impl FnOnce(BoundsProvenance) -> Result<T, BoundsError>,
) -> Result<T, BoundsError> {
  let Some(choices) = choices else {
    return run(BoundsProvenance::default());
  };
  let missing = Rc::new(Cell::new(false));
  let result = run(BoundsProvenance {
    choices: Some(ChoiceState::Refine {
      choices: Arc::clone(choices),
      active,
      missing: Rc::clone(&missing),
    }),
  })?;
  if missing.get() {
    Err(BoundsError::UndiscoveredChoice)
  } else {
    Ok(result)
  }
}

impl Aggregator<BoundsProvenance> for CountAggregator {
  fn aggregate(
    &self,
    prov: &BoundsProvenance,
    _: &RuntimeEnvironment,
    rows: DynamicElements<BoundsProvenance>,
  ) -> DynamicElements<BoundsProvenance> {
    count_rows(prov, rows, 0)
  }

  fn aggregate_implicit_group(
    &self,
    prov: &BoundsProvenance,
    _: &RuntimeEnvironment,
    rows: DynamicElements<BoundsProvenance>,
  ) -> DynamicElements<BoundsProvenance> {
    // An implicit key exists only in worlds where at least one member exists.
    count_rows(prov, rows, 1)
  }
}

impl Aggregator<BoundsProvenance> for ExistsAggregator {
  fn aggregate(
    &self,
    prov: &BoundsProvenance,
    _: &RuntimeEnvironment,
    rows: DynamicElements<BoundsProvenance>,
  ) -> DynamicElements<BoundsProvenance> {
    let exists = any_row(prov, rows);
    let absent = prov.negate(&exists).expect("bounds provenance supports negation");
    vec![DynamicElement::new(true, exists), DynamicElement::new(false, absent)]
  }

  fn aggregate_implicit_group(
    &self,
    prov: &BoundsProvenance,
    _: &RuntimeEnvironment,
    rows: DynamicElements<BoundsProvenance>,
  ) -> DynamicElements<BoundsProvenance> {
    vec![DynamicElement::new(true, any_row(prov, rows))]
  }
}

impl Aggregator<BoundsProvenance> for SumProdAggregator {
  fn aggregate(
    &self,
    prov: &BoundsProvenance,
    _: &RuntimeEnvironment,
    rows: DynamicElements<BoundsProvenance>,
  ) -> DynamicElements<BoundsProvenance> {
    numeric_rows(self, prov, rows, true)
  }

  fn aggregate_implicit_group(
    &self,
    prov: &BoundsProvenance,
    _: &RuntimeEnvironment,
    rows: DynamicElements<BoundsProvenance>,
  ) -> DynamicElements<BoundsProvenance> {
    // Unlike an explicit domain, a memberless implicit key has no output,
    // even when its nonempty subsets could also produce the identity.
    numeric_rows(self, prov, rows, false)
  }
}

fn numeric_rows(
  aggregate: &SumProdAggregator,
  prov: &BoundsProvenance,
  rows: DynamicElements<BoundsProvenance>,
  emit_empty: bool,
) -> DynamicElements<BoundsProvenance> {
  let identity = aggregate.perform_sum_prod(std::iter::empty());
  let mut empty = prov.one();
  // Equal totals share the rest of the fold, but may arise in different worlds.
  let mut totals = BTreeMap::<Tuple, BoundsTag>::new();
  let merge = |totals: &mut BTreeMap<_, BoundsTag>, tuple, tag| {
    if !prov.discard(&tag) {
      totals
        .entry(tuple)
        .and_modify(|prior| *prior = prov.add(prior, &tag))
        .or_insert(tag);
    }
  };
  for row in rows {
    let not_present = prov.negate(&row.tag).expect("bounds provenance supports negation");
    let input = aggregate.numeric_value(&row.tuple);
    let mut next = BTreeMap::new();
    merge(
      &mut next,
      aggregate.combine_numeric(&identity, input),
      prov.mult(&empty, &row.tag),
    );
    for (value, tag) in totals {
      merge(
        &mut next,
        aggregate.combine_numeric(&value, input),
        prov.mult(&tag, &row.tag),
      );
      merge(&mut next, value, prov.mult(&tag, &not_present));
    }
    empty = prov.mult(&empty, &not_present);
    totals = next;
  }
  if emit_empty {
    merge(&mut totals, identity, empty);
  }
  totals
    .into_iter()
    .map(|(value, tag)| DynamicElement::new(value, tag))
    .collect()
}

fn any_row(prov: &BoundsProvenance, rows: DynamicElements<BoundsProvenance>) -> BoundsTag {
  rows
    .into_iter()
    .fold(prov.zero(), |present, row| prov.add(&present, &row.tag))
}

fn count_rows(
  prov: &BoundsProvenance,
  rows: DynamicElements<BoundsProvenance>,
  first_count: usize,
) -> DynamicElements<BoundsProvenance> {
  let mut at_least = vec![prov.one()];
  for row in rows {
    at_least.push(prov.zero());
    for count in (1..at_least.len()).rev() {
      let with_row = prov.mult(&row.tag, &at_least[count - 1]);
      at_least[count] = prov.add(&at_least[count], &with_row);
    }
  }
  // Exactly k is "at least k" and not "at least k+1" in the same world.
  let impossible = prov.zero();
  at_least
    .iter()
    .zip(at_least.iter().skip(1).chain(std::iter::once(&impossible)))
    .enumerate()
    .skip(first_count)
    .filter_map(|(count, (enough, more))| {
      let tag = prov.minus(enough, more)?;
      (!prov.discard(&tag)).then(|| DynamicElement::new(count, tag))
    })
    .collect()
}

impl Provenance for BoundsProvenance {
  type InputTag = FactTag;
  type Tag = BoundsTag;
  type OutputTag = BoundsTag;

  fn name(&self) -> String {
    "anytimebounds".into()
  }

  fn tagging_fn(&self, tag: FactTag) -> BoundsTag {
    match tag {
      FactTag::Variable(id) => {
        let variable = DNFFormula::singleton(id);
        BoundsTag(variable.clone(), variable)
      }
      FactTag::Unknown => BoundsTag(DNFFormula::zero(), DNFFormula::one()),
    }
  }

  fn rule_choice_tagger(&self) -> Option<impl Fn(usize, Tuple, Tuple) -> BoundsTag + '_> {
    let choices = self.choices.as_ref()?;
    Some(move |rule, grounding, alt| match choices {
      ChoiceState::Discover(support) => {
        support.borrow_mut().entry((rule, grounding)).or_default().insert(alt);
        self.tagging_fn(FactTag::Unknown)
      }
      ChoiceState::Refine {
        choices,
        active,
        missing,
      } => match choices.get(&(rule, grounding)).and_then(|group| group.get(&alt)) {
        Some(ChoiceOutcome::Certain) => self.one(),
        Some(ChoiceOutcome::Symbolic { id, activation }) if activation <= active => {
          self.tagging_fn(FactTag::Variable(*id))
        }
        Some(ChoiceOutcome::Symbolic { .. }) => self.tagging_fn(FactTag::Unknown),
        None => {
          missing.set(true);
          self.tagging_fn(FactTag::Unknown)
        }
      },
    })
  }

  fn recover_fn(&self, tag: &BoundsTag) -> BoundsTag {
    tag.clone()
  }

  fn discard(&self, tag: &BoundsTag) -> bool {
    tag.1.is_empty()
  }

  fn zero(&self) -> BoundsTag {
    BoundsTag(DNFFormula::zero(), DNFFormula::zero())
  }

  fn one(&self) -> BoundsTag {
    BoundsTag(DNFFormula::one(), DNFFormula::one())
  }

  fn add(&self, left: &BoundsTag, right: &BoundsTag) -> BoundsTag {
    BoundsTag(minimize(left.0.or(&right.0)), minimize(left.1.or(&right.1)))
  }

  fn mult(&self, left: &BoundsTag, right: &BoundsTag) -> BoundsTag {
    BoundsTag(minimize(left.0.and(&right.0)), minimize(left.1.and(&right.1)))
  }

  fn negate(&self, tag: &BoundsTag) -> Option<BoundsTag> {
    let complement = |formula: &DNFFormula| {
      formula.iter().fold(DNFFormula::one(), |negated, clause| {
        let alternatives = DNFFormula::new(
          clause
            .iter()
            .map(|literal| Clause::singleton(literal.negate()))
            .collect(),
        );
        minimize(negated.and(&alternatives))
      })
    };
    Some(BoundsTag(complement(&tag.1), complement(&tag.0)))
  }

  fn saturated(&self, old: &BoundsTag, new: &BoundsTag) -> bool {
    old == new
  }
}

fn minimize(mut formula: DNFFormula) -> DNFFormula {
  fn subset(left: &Clause, right: &Clause) -> bool {
    let mut candidates = right.iter();
    left
      .iter()
      .all(|l| candidates.by_ref().find(|r| r.fact_id() >= l.fact_id()) == Some(l))
  }

  // Keep the minimal clauses of the discovered proofs so rediscovering a
  // proof, or a more restrictive version of it, cannot prolong recursion.
  formula
    .clauses
    .sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
  let mut minimal = Vec::new();
  for clause in formula.clauses {
    if !minimal.iter().any(|existing| subset(existing, &clause)) {
      minimal.push(clause);
    }
  }
  DNFFormula::new(minimal)
}

struct Rounded;

impl Semiring for Rounded {
  type Element = ProbabilityInterval;

  fn zero(&self) -> ProbabilityInterval {
    ProbabilityInterval::ZERO
  }

  fn one(&self) -> ProbabilityInterval {
    ProbabilityInterval::ONE
  }

  fn add(&self, a: ProbabilityInterval, b: ProbabilityInterval) -> ProbabilityInterval {
    if a == ProbabilityInterval::ZERO {
      return b;
    }
    if b == ProbabilityInterval::ZERO {
      return a;
    }
    enclosure(a.lower + b.lower, a.upper + b.upper)
  }

  fn mult(&self, a: ProbabilityInterval, b: ProbabilityInterval) -> ProbabilityInterval {
    if a == ProbabilityInterval::ZERO || b == ProbabilityInterval::ZERO {
      return ProbabilityInterval::ZERO;
    }
    if a == ProbabilityInterval::ONE {
      return b;
    }
    if b == ProbabilityInterval::ONE {
      return a;
    }
    enclosure(a.lower * b.lower, a.upper * b.upper)
  }

  fn negate(&self, a: ProbabilityInterval) -> ProbabilityInterval {
    if a == ProbabilityInterval::ZERO {
      return ProbabilityInterval::ONE;
    }
    if a == ProbabilityInterval::ONE {
      return ProbabilityInterval::ZERO;
    }
    enclosure(1.0 - a.upper, 1.0 - a.lower)
  }
}

fn enclosure(lower: f64, upper: f64) -> ProbabilityInterval {
  ProbabilityInterval {
    lower: lower.next_down().max(0.0),
    upper: upper.next_up().min(1.0),
  }
}
