use crate::common::tuple::*;

use super::*;

#[derive(Clone)]
pub struct DynamicJoinIndexedVecDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub indexed_vec: &'a DynamicIndexedVecCollection<Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicJoinIndexedVecDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.d1.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.d1.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicJoinIndexedVecDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let indexed_vec = self.indexed_vec;
    let ctx = self.ctx;
    batches.unary(move |batch: DynamicBatch<'a, Prov>| {
      DynamicBatch::from_iterator(batch.filter_map(move |DynamicElement { tuple, tag }| {
        let Tuple::Tuple(fields) = tuple else {
          panic!("Indexed join requires a left tuple");
        };
        let mut fields = fields.into_vec().into_iter();
        let key = fields.next().expect("Indexed join requires a left key");
        let (tail, right_tag) = indexed_vec.get_ref(key.as_usize())?;
        let tag = ctx.mult(&tag, right_tag);
        let payload = fields.next().expect("Indexed join requires a left payload");
        Some(DynamicElement::new((key, payload, tail.clone()), tag))
      }))
    })
  }
}
