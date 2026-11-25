use std::marker::PhantomData;

use super::*;
use crate::runtime::dynamic::dataflow::batching::{DynamicBatch, DynamicBatches};
use crate::runtime::dynamic::{dataflow as dynamic, DynamicElement};

/// Adapt a typed stream for a built-in operator shared with the interpreter.
pub fn to_dynamic<'a, Tup, Prov, D>(source: D) -> dynamic::DynamicDataflow<'a, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<Tup, Prov> + 'a,
  D::Stable: 'a,
  D::Recent: 'a,
  <D::Stable as Batches<Tup, Prov>>::Batch: 'a,
  <D::Recent as Batches<Tup, Prov>>::Batch: 'a,
{
  dynamic::DynamicDataflow::new(TypedSource(source, PhantomData))
}

#[derive(Clone)]
struct TypedSource<D, Tup>(D, PhantomData<Tup>);

impl<'a, D, Tup, Prov> dynamic::Dataflow<'a, Prov> for TypedSource<D, Tup>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<Tup, Prov> + 'a,
  D::Stable: 'a,
  D::Recent: 'a,
  <D::Stable as Batches<Tup, Prov>>::Batch: 'a,
  <D::Recent as Batches<Tup, Prov>>::Batch: 'a,
{
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::from_iterator(self.0.iter_stable().map(dynamic_batch))
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::from_iterator(self.0.clone().iter_recent().map(dynamic_batch))
  }
}

fn dynamic_batch<'a, Tup: StaticTupleTrait, Prov: Provenance>(
  batch: impl Batch<Tup, Prov> + 'a,
) -> DynamicBatch<'a, Prov> {
  DynamicBatch::from_iterator(batch.map(|element| {
    let (tuple, tag) = element.into();
    DynamicElement::new(tuple.into_dyn_tuple(), tag)
  }))
}

pub fn from_dynamic<Tup: StaticTupleTrait, Prov: Provenance>(
  source: dynamic::DynamicDataflow<'_, Prov>,
) -> BuiltinDataflow<'_, Tup, Prov> {
  BuiltinDataflow(source, PhantomData)
}

#[derive(Clone)]
pub struct BuiltinDataflow<'a, Tup, Prov: Provenance>(dynamic::DynamicDataflow<'a, Prov>, PhantomData<Tup>);

impl<'a, Tup: StaticTupleTrait, Prov: Provenance> Dataflow<Tup, Prov> for BuiltinDataflow<'a, Tup, Prov> {
  type Stable = BuiltinBatches<'a, Tup, Prov>;
  type Recent = BuiltinBatches<'a, Tup, Prov>;

  fn iter_stable(&self) -> Self::Stable {
    BuiltinBatches(dynamic::Dataflow::iter_stable(&self.0), PhantomData)
  }

  fn iter_recent(self) -> Self::Recent {
    BuiltinBatches(dynamic::Dataflow::iter_recent(&self.0), PhantomData)
  }
}

#[derive(Clone)]
pub struct BuiltinBatches<'a, Tup, Prov: Provenance>(DynamicBatches<'a, Prov>, PhantomData<Tup>);

impl<'a, Tup: StaticTupleTrait, Prov: Provenance> Iterator for BuiltinBatches<'a, Tup, Prov> {
  type Item = BuiltinBatch<'a, Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.0.next().map(|batch| BuiltinBatch(batch, PhantomData))
  }
}

impl<'a, Tup: StaticTupleTrait, Prov: Provenance> Batches<Tup, Prov> for BuiltinBatches<'a, Tup, Prov> {
  type Batch = BuiltinBatch<'a, Tup, Prov>;
}

#[derive(Clone)]
pub struct BuiltinBatch<'a, Tup, Prov: Provenance>(DynamicBatch<'a, Prov>, PhantomData<Tup>);

impl<Tup: StaticTupleTrait, Prov: Provenance> Iterator for BuiltinBatch<'_, Tup, Prov> {
  type Item = StaticElement<Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self
      .0
      .next()
      .map(|e| StaticElement::new(Tup::from_dyn_tuple(e.tuple), e.tag))
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> Batch<Tup, Prov> for BuiltinBatch<'_, Tup, Prov> {}
