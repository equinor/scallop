use std::cell::Ref;

use crate::common::tuple::*;
use crate::runtime::utils::gallop_index;

use super::batching::*;
use super::*;

#[derive(Clone)]
pub struct DynamicRelationDataflow<'a, Prov: Provenance>(pub &'a DynamicRelation<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicRelationDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicRelationStableBatches {
      collections: self.0.stable.borrow(),
      rela_id: 0,
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let rows = Ref::map(self.0.recent.borrow(), |collection| collection.elements.as_slice());
    DynamicBatches::single(DynamicRelationBatch { rows, elem_id: 0 })
  }
}

pub struct DynamicRelationStableBatches<'a, Prov: Provenance> {
  pub collections: Ref<'a, Vec<DynamicSortedCollection<Prov>>>,
  pub rela_id: usize,
}

impl<'a, Prov: Provenance> Clone for DynamicRelationStableBatches<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      collections: Ref::clone(&self.collections),
      rela_id: self.rela_id,
    }
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicRelationStableBatches<'a, Prov> {
  fn batch_size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.collections.len().saturating_sub(self.rela_id);
    (remaining, Some(remaining))
  }

  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    if self.rela_id < self.collections.len() {
      let rows = Ref::map(Ref::clone(&self.collections), |collections| {
        collections[self.rela_id].elements.as_slice()
      });
      self.rela_id += 1;
      Some(DynamicBatch::new(DynamicRelationBatch { rows, elem_id: 0 }))
    } else {
      None
    }
  }
}

struct DynamicRelationBatch<'a, Prov: Provenance> {
  rows: Ref<'a, [DynamicElement<Prov>]>,
  elem_id: usize,
}

impl<'a, Prov: Provenance> Clone for DynamicRelationBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      rows: Ref::clone(&self.rows),
      elem_id: self.elem_id,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicRelationBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    let elem = self.rows.get(self.elem_id)?;
    self.elem_id += 1;
    Some(elem.clone())
  }

  fn remaining_slice(&self) -> Option<&[DynamicElement<Prov>]> {
    Some(&self.rows[self.elem_id.min(self.rows.len())..])
  }

  fn is_tuple_sorted(&self) -> bool {
    true
  }

  fn is_materialized_and_seekable(&self) -> bool {
    true
  }

  fn step(&mut self, u: usize) {
    self.elem_id += u;
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.elem_id = gallop_index(&self.rows, self.elem_id, |row| &row.tuple < until);
    self.next_elem()
  }

  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.elem_id = gallop_index(&self.rows, self.elem_id, |row| &row.tuple[0] < until);
    self.next_elem()
  }
}
