use std::rc::Rc;

use crate::common::expr::*;
use crate::common::foreign_predicate::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateConstraintDataflow<'a, Prov: Provenance> {
  /// Sub-dataflow
  pub dataflow: DynamicDataflow<'a, Prov>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The arguments to the foreign predicate
  pub args: Rc<[Expr]>,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateConstraintDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(|| self.dataflow.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(|| self.dataflow.iter_recent())
  }
}

impl<'a, Prov: Provenance> ForeignPredicateConstraintDataflow<'a, Prov> {
  fn apply(&self, source: impl FnOnce() -> DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let fp = self
      .runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");
    let batches = source();
    let foreign_predicate = fp.clone();
    let args = Rc::clone(&self.args);
    let env = self.runtime;
    let ctx = self.ctx;
    batches.unary(move |batch: DynamicBatch<'a, Prov>| {
      let foreign_predicate = foreign_predicate.clone();
      let args = Rc::clone(&args);
      DynamicBatch::from_iterator(batch.filter_map(move |Tagged { tuple, tag }| {
        let values = args
          .iter()
          .map(|arg| match arg {
            Expr::Access(a) => tuple[a].as_value(),
            Expr::Constant(c) => c.clone(),
            _ => panic!("Invalid argument to bounded foreign predicate"),
          })
          .collect::<Vec<_>>();
        let result = foreign_predicate.evaluate_with_env(env, &values);
        if result.is_empty() {
          return None;
        }
        assert_eq!(
          result.len(),
          1,
          "Bounded foreign predicate should return at most one element per evaluation"
        );
        let input_tag = Prov::InputTag::from_dynamic_input_tag(&result[0].0);
        let new_tag = ctx.tagging_optional_fn(input_tag);
        Some(DynamicElement::new(tuple, ctx.mult(&tag, &new_tag)))
      }))
    })
  }
}
