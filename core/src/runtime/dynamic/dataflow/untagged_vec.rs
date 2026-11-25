use std::rc::Rc;

use crate::common::tuple::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DynamicUntaggedVec<'a, Prov: Provenance> {
  ctx: &'a Prov,
  tuples: Rc<Vec<Tuple>>,
}

impl<'a, Prov: Provenance> DynamicUntaggedVec<'a, Prov> {
  pub fn new(ctx: &'a Prov, tuples: Vec<Tuple>) -> Self {
    Self {
      ctx,
      tuples: Rc::new(tuples),
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicUntaggedVec<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let tuples = Rc::clone(&self.tuples);
    let ctx = self.ctx;
    let batch = DynamicBatch::from_iterator((0..tuples.len()).map(move |i| {
      let tuple = tuples[i].clone();
      DynamicElement::new(tuple, ctx.one())
    }));
    DynamicBatches::from_iterator(std::iter::once(batch))
  }
}
