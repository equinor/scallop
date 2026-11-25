use super::super::*;

pub trait BatchUnaryOp<'a, Prov: Provenance>: dyn_clone::DynClone + 'a {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov>;
}

dyn_clone::clone_trait_object!(<'a, Prov> BatchUnaryOp<'a, Prov> where Prov: Provenance);

impl<'a, Prov, F> BatchUnaryOp<'a, Prov> for F
where
  Prov: Provenance,
  F: Fn(DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> + Clone + 'a,
{
  fn apply(&self, batch: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    self(batch)
  }
}

#[derive(Clone)]
pub struct DynamicBatchUnaryOp<'a, Prov: Provenance>(Box<dyn BatchUnaryOp<'a, Prov> + 'a>);

impl<'a, Prov: Provenance> DynamicBatchUnaryOp<'a, Prov> {
  pub fn new<Op: BatchUnaryOp<'a, Prov>>(op: Op) -> Self {
    Self(Box::new(op))
  }
}

impl<'a, Prov: Provenance> BatchUnaryOp<'a, Prov> for DynamicBatchUnaryOp<'a, Prov> {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    self.0.apply(b)
  }
}
