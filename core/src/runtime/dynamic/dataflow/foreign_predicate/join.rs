use std::rc::Rc;

use crate::common::expr::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateJoinDataflow<'a, Prov: Provenance> {
  pub left: DynamicDataflow<'a, Prov>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub args: Rc<[Expr]>,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateJoinDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.left.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.left.iter_recent())
  }
}

impl<'a, Prov: Provenance> ForeignPredicateJoinDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let env = self.runtime;
    let ctx = self.ctx;
    let foreign_predicate = env
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found")
      .clone();
    let args = Rc::clone(&self.args);
    batches.unary(move |mut batch: DynamicBatch<'a, Prov>| {
      // Prefetch before cloning the predicate so the cursor starts with the updated state.
      let current_output_batch = batch
        .next_elem()
        .map(|elem| eval_foreign_predicate(elem, &foreign_predicate, &args, env, ctx));
      DynamicBatch::new(ForeignPredicateJoinBatch {
        batch,
        foreign_predicate: foreign_predicate.clone(),
        args: Rc::clone(&args),
        current_output_batch,
        env,
        ctx,
      })
    })
  }
}

#[derive(Clone)]
struct ForeignPredicateJoinBatch<'a, Prov: Provenance> {
  batch: DynamicBatch<'a, Prov>,
  foreign_predicate: DynamicForeignPredicate,
  args: Rc<[Expr]>,
  current_output_batch: Option<(DynamicElement<Prov>, std::vec::IntoIter<DynamicElement<Prov>>)>,
  env: &'a RuntimeEnvironment,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ForeignPredicateJoinBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some((left_elem, current_output_batch)) = &mut self.current_output_batch {
      match current_output_batch.next() {
        Some(right_elem) => {
          let tuple = (left_elem.tuple.clone(), right_elem.tuple);
          let new_tag = self.ctx.mult(&left_elem.tag, &right_elem.tag);
          return Some(DynamicElement::new(tuple, new_tag));
        }
        None => {
          self.current_output_batch = self
            .batch
            .next_elem()
            .map(|elem| eval_foreign_predicate(elem, &self.foreign_predicate, &self.args, self.env, self.ctx));
        }
      }
    }
    None
  }
}

/// Evaluate the foreign predicate on the given element
fn eval_foreign_predicate<Prov: Provenance>(
  elem: DynamicElement<Prov>,
  fp: &DynamicForeignPredicate,
  args: &[Expr],
  env: &RuntimeEnvironment,
  ctx: &Prov,
) -> (DynamicElement<Prov>, std::vec::IntoIter<DynamicElement<Prov>>) {
  // First get the arguments to pass to the foreign predicate
  let args_to_fp: Vec<Value> = args
    .iter()
    .map(|arg| match arg {
      Expr::Access(a) => elem.tuple[a].as_value(),
      Expr::Constant(c) => c.clone(),
      _ => panic!("Foreign predicate join only supports constant and access arguments"),
    })
    .collect();

  (elem, evaluate_output(fp, &args_to_fp, env, ctx).into_iter())
}
