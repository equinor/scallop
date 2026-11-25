use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DynamicRecentCollectionDataflow<'a, Prov: Provenance>(pub DynamicCollectionRef<'a, Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicRecentCollectionDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    collection_batches(&self.0)
  }
}

#[derive(Clone)]
pub struct DynamicStableCollectionDataflow<'a, Prov: Provenance>(pub DynamicCollectionRef<'a, Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicStableCollectionDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    collection_batches(&self.0)
  }
}

fn collection_batches<'a, Prov: Provenance>(collection: &DynamicCollectionRef<'a, Prov>) -> DynamicBatches<'a, Prov> {
  match collection {
    DynamicCollectionRef::Sorted(rows) => DynamicBatches::single(RefElementsBatch::new(&rows.elements)),
    DynamicCollectionRef::IndexedVec(_) => DynamicBatches::single(collection.iter()),
  }
}

#[derive(Clone)]
pub struct DynamicRecentSortedCollectionDataflow<'a, Prov: Provenance>(pub &'a DynamicSortedCollection<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicRecentSortedCollectionDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(RefElementsBatch::new(&self.0.elements))
  }
}

#[derive(Clone)]
pub struct DynamicStableSortedCollectionDataflow<'a, Prov: Provenance>(pub &'a DynamicSortedCollection<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicStableSortedCollectionDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(RefElementsBatch::new(&self.0.elements))
  }
}
