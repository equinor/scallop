use super::*;
use crate::runtime::dynamic::{DynamicElement, DynamicIndexedVecCollection, DynamicSortedCollection};
use crate::runtime::env::RuntimeEnvironment;
use crate::runtime::provenance::*;
use crate::runtime::utils::merge_sorted_tagged;

#[derive(Clone)]
pub struct StaticCollection<Tup: StaticTupleTrait, Prov: Provenance> {
  pub elements: Vec<StaticElement<Tup, Prov>>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> std::fmt::Debug for StaticCollection<Tup, Prov>
where
  Prov::Tag: std::fmt::Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(&self.elements).finish()
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> std::fmt::Display for StaticCollection<Tup, Prov>
where
  StaticElement<Tup, Prov>: std::fmt::Display,
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

impl<Tup: StaticTupleTrait, Prov: Provenance> StaticCollection<Tup, Prov> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }

  pub fn from_vec_unchecked(elements: Vec<StaticElement<Tup, Prov>>) -> Self {
    Self { elements }
  }

  pub fn from_vec(mut elements: Vec<StaticElement<Tup, Prov>>, ctx: &Prov) -> Self {
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

  pub fn ith(&self, i: usize) -> Option<&StaticElement<Tup, Prov>> {
    self.elements.get(i)
  }

  pub fn iter(&self) -> impl Iterator<Item = &StaticElement<Tup, Prov>> {
    self.elements.iter()
  }

  pub fn recover(self, ctx: &Prov) -> StaticOutputCollection<Tup, Prov> {
    StaticOutputCollection::from(
      self
        .elements
        .into_iter()
        .map(move |elem| (ctx.recover_fn(&elem.tag), elem.tuple.get().clone())),
    )
  }

  pub fn recover_with_env(self, ctx: &Prov, env: &RuntimeEnvironment) -> StaticOutputCollection<Tup, Prov> {
    StaticOutputCollection::from(self.elements.into_iter().filter_map(|elem| {
      let (tuple, tag) = elem.into();
      let tuple = env.externalize_tuple(&tuple.into_dyn_tuple())?;
      Some((ctx.recover_fn(&tag), Tup::from_dyn_tuple(tuple)))
    }))
  }

  pub(crate) fn into_dynamic_sorted(self) -> DynamicSortedCollection<Prov> {
    DynamicSortedCollection::from_vec_unchecked(
      self
        .into_iter()
        .map(|elem| {
          let (tuple, tag) = elem.into();
          DynamicElement::new(tuple.into_dyn_tuple(), tag)
        })
        .collect(),
    )
  }

  pub(crate) fn from_dynamic_sorted(collection: DynamicSortedCollection<Prov>) -> Self {
    Self::from_dynamic_elements(collection)
  }

  pub fn from_dynamic_elements(collection: impl IntoIterator<Item = DynamicElement<Prov>>) -> Self {
    Self::from_vec_unchecked(
      collection
        .into_iter()
        .map(|e| StaticElement::new(Tup::from_dyn_tuple(e.tuple), e.tag))
        .collect(),
    )
  }

  pub fn into_indexed_vec(self, ctx: &Prov) -> DynamicIndexedVecCollection<Prov> {
    DynamicIndexedVecCollection::from_vec(self.into_dynamic_sorted().elements, ctx)
  }

  pub fn merge(self, other: Self, ctx: &Prov) -> Self {
    Self {
      elements: merge_sorted_tagged(self.elements, other.elements, ctx),
    }
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> IntoIterator for StaticCollection<Tup, Prov> {
  type Item = StaticElement<Tup, Prov>;
  type IntoIter = std::vec::IntoIter<Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.elements.into_iter()
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> std::ops::Deref for StaticCollection<Tup, Prov> {
  type Target = [StaticElement<Tup, Prov>];

  fn deref(&self) -> &Self::Target {
    &self.elements[..]
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> std::ops::DerefMut for StaticCollection<Tup, Prov> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.elements[..]
  }
}
