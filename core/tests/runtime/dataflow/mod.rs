mod dyn_aggregate;
mod dyn_collection;
mod dyn_difference;
mod dyn_exclusion;
mod dyn_filter;
mod dyn_find;
mod dyn_foreign_predicate;
mod dyn_group_aggregate;
mod dyn_group_by_key;
mod dyn_intersect;
mod dyn_join;
mod dyn_product;
mod dyn_project;
mod dyn_relation;

use std::cell::Cell;

use scallop_core::runtime::dynamic::dataflow::batching::{Batch, DynamicBatches};
use scallop_core::runtime::dynamic::dataflow::Dataflow;
use scallop_core::runtime::dynamic::DynamicElement;
use scallop_core::runtime::provenance::Provenance;

#[derive(Clone)]
struct CountedBatch<'a, P: Provenance> {
  rows: std::slice::Iter<'a, DynamicElement<P>>,
  reads: &'a Cell<usize>,
  contiguous: bool,
}

impl<'a, P: Provenance> CountedBatch<'a, P> {
  fn new(rows: &'a [DynamicElement<P>], reads: &'a Cell<usize>) -> Self {
    Self {
      rows: rows.iter(),
      reads,
      contiguous: false,
    }
  }

  fn contiguous(self) -> Self {
    Self {
      contiguous: true,
      ..self
    }
  }
}

impl<'a, P: Provenance> Batch<'a, P> for CountedBatch<'a, P> {
  fn next_elem(&mut self) -> Option<DynamicElement<P>> {
    let row = self.rows.next()?;
    self.reads.set(self.reads.get() + 1);
    Some(row.clone())
  }

  fn remaining_slice(&self) -> Option<&[DynamicElement<P>]> {
    self.contiguous.then_some(self.rows.as_slice())
  }
}

impl<'a, P: Provenance> Dataflow<'a, P> for CountedBatch<'a, P> {
  fn iter_recent(&self) -> DynamicBatches<'a, P> {
    DynamicBatches::single(self.clone())
  }
}
