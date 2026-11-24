use super::*;
use crate::common::expr::Expr;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub expression: Expr,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicProjectDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicProjectDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let runtime = self.runtime;
    let expression = self.expression.clone();
    DynamicBatches::from_iterator(batches.map(move |batch| {
      let expression = expression.clone();
      DynamicBatch::from_iterator(batch.filter_map(move |elem| {
        let tuple = runtime.eval(&expression, &elem.tuple)?;
        Some(DynamicElement::new(tuple, elem.tag))
      }))
    }))
  }
}
