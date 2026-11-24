use super::*;
use crate::common::expr::Expr;

#[derive(Clone)]
pub struct DynamicFilterDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub filter: Expr,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicFilterDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicFilterDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let runtime = self.runtime;
    let filter = self.filter.clone();
    DynamicBatches::from_iterator(batches.map(move |batch| {
      let filter = filter.clone();
      batch.filter(move |elem| runtime.eval(&filter, &elem.tuple).is_some_and(|tup| tup.as_bool()))
    }))
  }
}
