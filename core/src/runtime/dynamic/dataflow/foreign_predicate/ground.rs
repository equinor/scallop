use super::*;

#[derive(Clone)]
pub struct ForeignPredicateGroundDataflow<'a, Prov: Provenance> {
  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub bounded_constants: Vec<Value>,

  /// Whether this Dataflow is running on first iteration
  pub first_iteration: bool,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> ForeignPredicateGroundDataflow<'a, Prov> {
  /// Generate a batch from the foreign predicate
  fn generate_batch(&self) -> ElementsBatch<Prov> {
    // Fetch the foreign predicate
    let foreign_predicate = self
      .runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");
    ElementsBatch::new(evaluate_output(
      foreign_predicate,
      &self.bounded_constants,
      self.runtime,
      self.ctx,
    ))
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateGroundDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::empty()
    } else {
      DynamicBatches::single(self.generate_batch())
    }
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::single(self.generate_batch())
    } else {
      DynamicBatches::empty()
    }
  }
}
