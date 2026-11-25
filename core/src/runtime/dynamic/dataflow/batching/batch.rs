use crate::common::tuple::Tuple;
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::gallop_index;

pub trait Batch<'a, Prov: Provenance>: 'a + dyn_clone::DynClone {
  /// Get the next element in the batch
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>>;

  /// Borrow all unread elements, in iteration order, without advancing this batch.
  fn remaining_slice(&self) -> Option<&[DynamicElement<Prov>]> {
    None
  }

  /// Transfer an owned buffer of unread rows, leaving the batch exhausted.
  /// Return `None` without advancing if there is no buffer to transfer.
  fn take_materialized(&mut self) -> Option<DynamicElements<Prov>> {
    None
  }

  /// Whether this batch yields tuples in nondecreasing order.
  fn is_tuple_sorted(&self) -> bool {
    false
  }

  /// Whether this batch already holds sorted rows and supports both seek methods
  /// and `remaining_slice` without materializing another buffer.
  fn is_materialized_and_seekable(&self) -> bool {
    false
  }

  /// Step the batch iterator for `u` steps
  fn step(&mut self, u: usize) {
    for _ in 0..u {
      self.next_elem();
    }
  }

  /// On batches that support seeking, return the first unread tuple at or above the bound.
  /// Other batches just advance one row.
  fn search_until(&mut self, _until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.next_elem()
  }

  /// Like `search_until`, but compare only the first tuple field.
  fn search_elem_0_until(&mut self, _until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.next_elem()
  }
}

dyn_clone::clone_trait_object!(<'a, Prov> Batch<'a, Prov> where Prov: Provenance);

#[derive(Clone)]
pub struct DynamicBatch<'a, Prov: Provenance>(Box<dyn Batch<'a, Prov>>);

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.0.next_elem()
  }

  fn remaining_slice(&self) -> Option<&[DynamicElement<Prov>]> {
    self.0.remaining_slice()
  }

  fn take_materialized(&mut self) -> Option<DynamicElements<Prov>> {
    self.0.take_materialized()
  }

  fn is_tuple_sorted(&self) -> bool {
    self.0.is_tuple_sorted()
  }

  fn is_materialized_and_seekable(&self) -> bool {
    self.0.is_materialized_and_seekable()
  }

  fn step(&mut self, u: usize) {
    self.0.step(u)
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.0.search_until(until)
  }

  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.0.search_elem_0_until(until)
  }
}

impl<'a, Prov: Provenance> DynamicBatch<'a, Prov> {
  pub fn new<B: Batch<'a, Prov>>(b: B) -> Self {
    Self(Box::new(b))
  }

  pub fn from_iterator(iter: impl Iterator<Item = DynamicElement<Prov>> + Clone + 'a) -> Self {
    Self::new(IteratorBatch(iter))
  }

  pub fn filter<F: Fn(&DynamicElement<Prov>) -> bool + Clone + 'a>(self, f: F) -> Self {
    Self::new(FilterBatch {
      source: self,
      predicate: f,
    })
  }

  pub fn collect_vec(&mut self) -> Vec<DynamicElement<Prov>> {
    self.collect()
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.next_elem()
  }
}

#[derive(Clone)]
pub struct RefElementsBatch<'a, Prov: Provenance> {
  iter: std::slice::Iter<'a, DynamicElement<Prov>>,
  sorted: bool,
}

impl<'a, Prov: Provenance> RefElementsBatch<'a, Prov> {
  pub fn new(elements: &'a DynamicElements<Prov>) -> Self {
    Self {
      iter: elements.iter(),
      sorted: false,
    }
  }

  pub(crate) fn new_sorted(elements: &'a DynamicSortedCollection<Prov>) -> Self {
    Self {
      iter: elements.elements.iter(),
      sorted: true,
    }
  }

  fn seek(&mut self, mut less: impl FnMut(&Tuple) -> bool) -> Option<DynamicElement<Prov>> {
    if self.sorted {
      let rows = self.iter.as_slice();
      let start = gallop_index(rows, 0, |row| less(&row.tuple));
      self.iter = rows[start..].iter();
    }
    self.next_elem()
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for RefElementsBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.iter.next().cloned()
  }

  fn remaining_slice(&self) -> Option<&[DynamicElement<Prov>]> {
    Some(self.iter.as_slice())
  }

  fn is_tuple_sorted(&self) -> bool {
    self.sorted
  }

  fn is_materialized_and_seekable(&self) -> bool {
    self.sorted
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.seek(|tuple| tuple < until)
  }

  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.seek(|tuple| &tuple[0] < until)
  }
}

#[derive(Clone)]
struct FilterBatch<'a, Prov: Provenance, F> {
  source: DynamicBatch<'a, Prov>,
  predicate: F,
}

impl<'a, Prov: Provenance, F> Batch<'a, Prov> for FilterBatch<'a, Prov, F>
where
  F: Fn(&DynamicElement<Prov>) -> bool + Clone + 'a,
{
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.source.find(&self.predicate)
  }

  fn is_tuple_sorted(&self) -> bool {
    self.source.is_tuple_sorted()
  }
}

#[derive(Clone)]
pub struct ElementsBatch<Prov: Provenance> {
  iter: std::vec::IntoIter<DynamicElement<Prov>>,
}

impl<Prov: Provenance> ElementsBatch<Prov> {
  pub fn new(elems: DynamicElements<Prov>) -> Self {
    Self {
      iter: elems.into_iter(),
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ElementsBatch<Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.iter.next()
  }

  fn take_materialized(&mut self) -> Option<DynamicElements<Prov>> {
    Some(std::mem::take(&mut self.iter).collect())
  }
}

#[derive(Clone)]
struct IteratorBatch<I>(I);

impl<'a, Prov, I> Batch<'a, Prov> for IteratorBatch<I>
where
  Prov: Provenance,
  I: Iterator<Item = DynamicElement<Prov>> + Clone + 'a,
{
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.0.next()
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicCollectionIter<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    match self {
      Self::Sorted(s) => s.next(),
      Self::IndexedVec(s) => s.next(),
    }
  }
}
