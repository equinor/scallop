use super::super::*;

pub trait BatchBinaryOp<'a, Prov: Provenance>: dyn_clone::DynClone + 'a {
  fn apply(&self, b1: DynamicBatch<'a, Prov>, b2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov>;
}

dyn_clone::clone_trait_object!(<'a, Prov> BatchBinaryOp<'a, Prov> where Prov: Provenance);

#[derive(Clone)]
pub struct DynamicBatchBinaryOp<'a, Prov: Provenance>(Box<dyn BatchBinaryOp<'a, Prov> + 'a>);

impl<'a, Prov: Provenance> DynamicBatchBinaryOp<'a, Prov> {
  pub fn new<Op: BatchBinaryOp<'a, Prov>>(op: Op) -> Self {
    Self(Box::new(op))
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for DynamicBatchBinaryOp<'a, Prov> {
  fn apply(&self, b1: DynamicBatch<'a, Prov>, b2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    self.0.apply(b1, b2)
  }
}
