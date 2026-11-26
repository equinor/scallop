use std::collections::*;

use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::compiler::ram::RuleChoice;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

type VisitedExclusionMap = <RcFamily as PointerFamily>::RcCell<HashMap<Tuple, usize>>;

#[derive(Clone)]
pub struct DynamicExclusionDataflow<'a, Prov: Provenance> {
  // The left dataflow which generates base tuples
  pub left: DynamicDataflow<'a, Prov>,

  // The right dataflow which generates tuples for exclusion
  // Current assumption is that the right dataflow is an `UntaggedVec`
  pub right: DynamicDataflow<'a, Prov>,

  // The provenance context
  pub ctx: &'a Prov,

  // The runtime environment
  pub runtime: &'a RuntimeEnvironment,

  choice: Option<&'a RuleChoice>,

  // A map of tuples that have been visited to their mutual exclusion IDs
  visited: VisitedExclusionMap,
}

impl<'a, Prov: Provenance> DynamicExclusionDataflow<'a, Prov> {
  pub fn new(
    left: DynamicDataflow<'a, Prov>,
    right: DynamicDataflow<'a, Prov>,
    ctx: &'a Prov,
    runtime: &'a RuntimeEnvironment,
  ) -> Self {
    Self {
      left,
      right,
      ctx,
      runtime,
      choice: None,
      visited: RcFamily::new_rc_cell(HashMap::new()),
    }
  }

  pub(crate) fn with_rule_choice(mut self, choice: Option<&'a RuleChoice>) -> Self {
    self.choice = choice;
    self
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicExclusionDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let left = self.left.iter_recent();
    let right = self.right.iter_recent();
    let op = ExclusionOp::new(
      self.runtime,
      self.ctx,
      RcFamily::clone_rc_cell(&self.visited),
      self.choice,
    );
    DynamicBatches::binary(left, right, op)
  }

  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let left = self.left.iter_stable();
    let right = self.right.iter_recent();
    let op = ExclusionOp::new(
      self.runtime,
      self.ctx,
      RcFamily::clone_rc_cell(&self.visited),
      self.choice,
    );
    DynamicBatches::binary(left, right, op)
  }
}

#[derive(Clone)]
pub struct ExclusionOp<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
  pub visited_exclusion_map: VisitedExclusionMap,
  choice: Option<&'a RuleChoice>,
}

impl<'a, Prov: Provenance> ExclusionOp<'a, Prov> {
  pub fn new(
    runtime: &'a RuntimeEnvironment,
    ctx: &'a Prov,
    visited_exclusion_map: VisitedExclusionMap,
    choice: Option<&'a RuleChoice>,
  ) -> Self {
    Self {
      runtime,
      ctx,
      visited_exclusion_map,
      choice,
    }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for ExclusionOp<'a, Prov> {
  fn apply(&self, left: DynamicBatch<'a, Prov>, right: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    DynamicBatch::new(DynamicExclusionBatch::new(
      self.runtime,
      self.ctx,
      RcFamily::clone_rc_cell(&self.visited_exclusion_map),
      left,
      right,
      self.choice,
    ))
  }
}

#[derive(Clone)]
pub struct DynamicExclusionBatch<'a, Prov: Provenance> {
  // Basic information
  pub runtime: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
  pub visited_exclusion_map: VisitedExclusionMap,
  choice: Option<&'a RuleChoice>,

  // Batches
  pub left: DynamicBatch<'a, Prov>,
  pub left_curr: Option<DynamicElement<Prov>>,
  pub right_source: DynamicBatch<'a, Prov>,
  pub right_clone: DynamicBatch<'a, Prov>,
  pub curr_exclusion_id: Option<usize>,
}

impl<'a, Prov: Provenance> DynamicExclusionBatch<'a, Prov> {
  pub fn new(
    runtime: &'a RuntimeEnvironment,
    ctx: &'a Prov,
    visited_exclusion_map: VisitedExclusionMap,
    mut left: DynamicBatch<'a, Prov>,
    right: DynamicBatch<'a, Prov>,
    choice: Option<&'a RuleChoice>,
  ) -> Self {
    let right_clone = right.clone();
    let left_curr = left.next_elem();
    Self {
      runtime,
      ctx,
      visited_exclusion_map,
      choice,
      left,
      left_curr,
      right_source: right,
      right_clone,
      curr_exclusion_id: None,
    }
  }

  fn legacy_id(&self, grounding: &Tuple) -> usize {
    if let Some(id) = RcFamily::get_rc_cell(&self.visited_exclusion_map, |m| m.get(grounding).cloned()) {
      id
    } else if let Some(id) = self.curr_exclusion_id {
      id
    } else {
      let id = self.runtime.allocate_new_exclusion_id();
      RcFamily::get_rc_cell_mut(&self.visited_exclusion_map, |m| m.insert(grounding.clone(), id));
      id
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicExclusionBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    let tagged_choice = self
      .choice
      .and_then(|choice| self.ctx.rule_choice_tagger().map(|tagger| (choice, tagger)));
    loop {
      if let Some(left) = &self.left_curr {
        let legacy_id = tagged_choice.is_none().then(|| self.legacy_id(&left.tuple));
        match self.right_clone.next_elem() {
          Some(right) => {
            let tuple: Tuple = (left.tuple.clone(), right.tuple).into();
            let me_tag = if let Some((choice, tagger)) = &tagged_choice {
              let alternative = self
                .runtime
                .eval(&choice.head, &tuple)
                .expect("rule choice head is a tuple projection");
              let grounding = Tuple::tuple(choice.variable_positions.iter().map(|&i| alternative[i].clone()));
              tagger(choice.rule, grounding, alternative)
            } else {
              let id = legacy_id.expect("legacy exclusion ID is allocated before reading the right batch");
              self
                .ctx
                .tagging_optional_fn(Prov::InputTag::from_dynamic_input_tag(&DynamicInputTag::Exclusive(id)))
            };
            let tag = self.ctx.mult(&left.tag, &me_tag);
            return Some(DynamicElement::new(tuple, tag));
          }
          None => {
            // Move on to the next left element and reset other states
            self.left_curr = self.left.next_elem();
            self.right_clone = self.right_source.clone();
            self.curr_exclusion_id = None;
          }
        }
      } else {
        return None;
      }
    }
  }
}
