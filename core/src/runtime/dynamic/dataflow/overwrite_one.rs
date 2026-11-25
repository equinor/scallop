use super::*;

#[derive(Clone)]
pub struct DynamicOverwriteOneDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicOverwriteOneDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicOverwriteOneDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let ctx = self.ctx;
    batches.unary(move |batch: DynamicBatch<'a, Prov>| {
      DynamicBatch::from_iterator(batch.map(move |mut elem| {
        elem.tag = ctx.one();
        elem
      }))
    })
  }
}
