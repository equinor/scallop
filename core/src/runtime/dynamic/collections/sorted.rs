use super::*;
use crate::common::tuple::Tuple;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;
use crate::runtime::utils::{gallop_index, merge_sorted_tagged};

#[derive(Clone)]
pub struct DynamicSortedCollection<Prov: Provenance> {
  pub elements: Vec<DynamicElement<Prov>>,
}

impl<Prov: Provenance> From<DynamicSortedCollection<Prov>> for DynamicCollection<Prov> {
  fn from(val: DynamicSortedCollection<Prov>) -> Self {
    DynamicCollection::Sorted(val)
  }
}

impl<Prov: Provenance> std::fmt::Debug for DynamicSortedCollection<Prov>
where
  Prov::Tag: std::fmt::Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(&self.elements).finish()
  }
}

impl<Prov: Provenance> std::fmt::Display for DynamicSortedCollection<Prov>
where
  DynamicElement<Prov>: std::fmt::Display,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, elem) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      std::fmt::Display::fmt(elem, f)?;
    }
    f.write_str("}")
  }
}

pub type DynamicSortedCollectionIter<'a, Prov> = core::iter::Cloned<core::slice::Iter<'a, DynamicElement<Prov>>>;

pub type DynamicSortedCollectionIntoIter<Prov> = <Vec<DynamicElement<Prov>> as IntoIterator>::IntoIter;

impl<Prov: Provenance> IntoIterator for DynamicSortedCollection<Prov> {
  type Item = DynamicElement<Prov>;
  type IntoIter = DynamicSortedCollectionIntoIter<Prov>;

  fn into_iter(self) -> Self::IntoIter {
    self.elements.into_iter()
  }
}

pub type DynamicSortedCollectionDrainer<'a, Prov> = std::vec::Drain<'a, DynamicElement<Prov>>;

impl<Prov: Provenance> DynamicSortedCollection<Prov> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }

  pub fn from_vec_unchecked(elements: Vec<DynamicElement<Prov>>) -> Self {
    Self { elements }
  }

  pub fn from_vec(mut elements: Vec<DynamicElement<Prov>>, ctx: &Prov) -> Self {
    elements.sort();
    elements.dedup_by(|later, kept| {
      if later.tuple != kept.tuple {
        return false;
      }
      kept.tag = ctx.add(&kept.tag, &later.tag);
      true
    });
    Self { elements }
  }

  pub fn len(&self) -> usize {
    self.elements.len()
  }

  pub fn is_empty(&self) -> bool {
    self.elements.is_empty()
  }

  pub fn ith(&self, i: usize) -> Option<&DynamicElement<Prov>> {
    self.elements.get(i)
  }

  pub fn iter<'a>(&'a self) -> DynamicSortedCollectionIter<'a, Prov> {
    self.elements.iter().cloned()
  }

  pub fn drain<'a>(&'a mut self) -> DynamicSortedCollectionDrainer<'a, Prov> {
    self.elements.drain(..)
  }

  /// Retain only the elements which the `retain_fn` returns true.
  ///
  /// The predicate receives each tagged element and its original index.
  /// This is an O(n) algorithm that preserves the ordering.
  pub fn retain<F>(&mut self, mut retain_fn: F)
  where
    F: FnMut(usize, &DynamicElement<Prov>) -> bool,
  {
    let mut index = 0;
    self.elements.retain(|element| {
      let keep = retain_fn(index, element);
      index += 1;
      keep
    });
  }

  pub fn apply_recover_fn<F, S>(self, mut f: F) -> impl Iterator<Item = (S, Tuple)>
  where
    F: FnMut(Prov::Tag) -> S,
  {
    self.elements.into_iter().map(move |elem| (f(elem.tag), elem.tuple))
  }

  pub fn recover(self, ctx: &Prov) -> DynamicOutputCollection<Prov> {
    DynamicOutputCollection::from(
      self
        .elements
        .into_iter()
        .map(move |elem| (ctx.recover_fn(&elem.tag), elem.tuple)),
    )
  }

  pub fn recover_with_monitor<M>(self, ctx: &Prov, m: &M) -> DynamicOutputCollection<Prov>
  where
    M: Monitor<Prov>,
  {
    DynamicOutputCollection::from(self.elements.into_iter().map(move |elem| {
      let output_tag = ctx.recover_fn(&elem.tag);
      m.observe_recover(&elem.tuple, &elem.tag, &output_tag);
      (output_tag, elem.tuple)
    }))
  }

  /// Insert the element into the collection
  pub fn insert(&mut self, elem: DynamicElement<Prov>, ctx: &Prov) {
    let to_insert_index = gallop_index(&self.elements, 0, |e| e.tuple < elem.tuple);
    if let Some(next_elem) = self.elements.get(to_insert_index) {
      if next_elem.tuple != elem.tuple {
        // If the tuple is not the same, we insert our new element
        self.elements.insert(to_insert_index, elem);
      } else {
        // If the tuple is the same, we update its tag
        self.elements[to_insert_index].tag = ctx.add(&elem.tag, &next_elem.tag);
      }
    } else {
      // If there does not exist an element that is greater than the element,
      // we push the element to the end
      self.elements.push(elem);
    }
  }

  /// Insert the element into the collection with the assumption that the element
  /// does not pre-exist in the collection.
  pub fn insert_unchecked(&mut self, elem: DynamicElement<Prov>) {
    let to_insert_index = gallop_index(&self.elements, 0, |e| e.tuple < elem.tuple);
    self.elements.insert(to_insert_index, elem);
  }

  pub fn merge(self, other: Self, ctx: &Prov) -> Self {
    Self {
      elements: merge_sorted_tagged(self.elements, other.elements, ctx),
    }
  }
}

impl<Prov: Provenance> std::ops::Deref for DynamicSortedCollection<Prov> {
  type Target = [DynamicElement<Prov>];

  fn deref(&self) -> &Self::Target {
    &self.elements[..]
  }
}

impl<Prov: Provenance> std::ops::DerefMut for DynamicSortedCollection<Prov> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.elements[..]
  }
}

impl<Prov: Provenance> std::ops::Index<usize> for DynamicSortedCollection<Prov> {
  type Output = DynamicElement<Prov>;

  fn index(&self, index: usize) -> &Self::Output {
    &self.elements[index]
  }
}

impl<Prov: Provenance> std::ops::IndexMut<usize> for DynamicSortedCollection<Prov> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.elements[index]
  }
}
