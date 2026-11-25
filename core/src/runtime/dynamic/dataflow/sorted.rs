use super::*;
use crate::common::tuple::Tuple;
use crate::runtime::utils::gallop_index;

#[derive(Clone)]
pub struct DynamicSortedDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicSortedDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.source.iter_stable().unary(SortOp)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.source.iter_recent().unary(SortOp)
  }
}

#[derive(Clone)]
pub struct SortOp;

impl<'a, Prov: Provenance> BatchUnaryOp<'a, Prov> for SortOp {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    if b.is_materialized_and_seekable() {
      b
    } else {
      DynamicBatch::new(DynamicSortedBatch::new(b))
    }
  }
}

#[derive(Clone)]
pub struct DynamicSortedBatch<'a, Prov: Provenance> {
  src: DynamicBatch<'a, Prov>,
  sorted_tgt: Option<std::vec::IntoIter<DynamicElement<Prov>>>,
}

impl<'a, Prov: Provenance> DynamicSortedBatch<'a, Prov> {
  pub fn new(src: DynamicBatch<'a, Prov>) -> Self {
    Self { src, sorted_tgt: None }
  }

  fn materialize(&mut self) -> Option<&mut std::vec::IntoIter<DynamicElement<Prov>>> {
    if self.sorted_tgt.is_none() {
      let sorted = self.src.is_tuple_sorted();
      let mut rows = self.src.take_materialized().unwrap_or_else(|| self.src.collect_vec());
      if rows.is_empty() {
        return None;
      }
      if !sorted {
        rows.sort();
      }
      self.sorted_tgt = Some(rows.into_iter());
    }
    self.sorted_tgt.as_mut()
  }

  fn seek(&mut self, mut less: impl FnMut(&Tuple) -> bool) -> Option<DynamicElement<Prov>> {
    let rows = self.materialize()?;
    let start = gallop_index(rows.as_slice(), 0, |row| less(&row.tuple));
    rows.nth(start)
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicSortedBatch<'a, Prov> {
  fn remaining_slice(&self) -> Option<&[DynamicElement<Prov>]> {
    self.sorted_tgt.as_ref().map(std::vec::IntoIter::as_slice)
  }

  fn take_materialized(&mut self) -> Option<DynamicElements<Prov>> {
    self.sorted_tgt.as_mut().map(|rows| std::mem::take(rows).collect())
  }

  fn is_tuple_sorted(&self) -> bool {
    true
  }

  fn is_materialized_and_seekable(&self) -> bool {
    self.sorted_tgt.is_some()
  }

  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.materialize()?.next()
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.seek(|tuple| tuple < until)
  }

  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.seek(|tuple| &tuple[0] < until)
  }
}
