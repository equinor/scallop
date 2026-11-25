use super::*;
use crate::common::tuple::Tuple;

#[derive(Clone)]
pub struct DynamicFindDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicFindDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicFindDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let key = self.key.clone();
    batches.unary(move |mut batch: DynamicBatch<'a, Prov>| {
      let curr_elem = batch.next_elem();
      DynamicBatch::new(DynamicFindBatch {
        source: batch,
        curr_elem,
        key: key.clone(),
      })
    })
  }
}

#[derive(Clone)]
struct DynamicFindBatch<'a, Prov: Provenance> {
  source: DynamicBatch<'a, Prov>,
  curr_elem: Option<DynamicElement<Prov>>,
  key: Tuple,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicFindBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    let key = &self.key;
    loop {
      let elem = self.curr_elem.as_ref()?;
      match elem.tuple[0].cmp(key) {
        Ordering::Less => self.curr_elem = self.source.search_elem_0_until(key),
        Ordering::Equal => return std::mem::replace(&mut self.curr_elem, self.source.next_elem()),
        Ordering::Greater => return None,
      }
    }
  }

  fn is_tuple_sorted(&self) -> bool {
    self.source.is_tuple_sorted()
  }
}
