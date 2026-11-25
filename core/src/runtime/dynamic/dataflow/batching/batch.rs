use crate::common::tuple::Tuple;
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

pub trait Batch<'a, Prov: Provenance>: 'a + dyn_clone::DynClone {
  /// Get the next element in the batch
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>>;

  /// Step the batch iterator for `u` steps
  fn step(&mut self, u: usize) {
    for _ in 0..u {
      self.next_elem();
    }
  }

  /// Search the elements until (before) the `until` tuple is reached;
  /// Should only have effect when the input is sorted.
  #[allow(unused)]
  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.next_elem()
  }

  /// Search the elements (using the first part `[0]` of the tuple) until (before) the `until` tuple is reached;
  /// Should only have effect when the input is sorted.
  #[allow(unused)]
  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
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

  fn step(&mut self, u: usize) {
    self.0.step(u)
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    self.0.search_until(until)
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
    Self::from_iterator(Iterator::filter(self, f))
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
}

impl<'a, Prov: Provenance> RefElementsBatch<'a, Prov> {
  pub fn new(elements: &'a DynamicElements<Prov>) -> Self {
    Self { iter: elements.iter() }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for RefElementsBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.iter.next().cloned()
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
