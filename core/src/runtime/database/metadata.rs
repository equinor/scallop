use std::str::FromStr;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

/// The metadata for the storage of a relation
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum StorageMetadata {
  /// Sorted Tuple Storage
  ///
  /// The default in-memory storage
  #[default]
  Sorted,

  /// Indexed Vector Storage:
  ///
  /// The first element of the tuple is the index in a dense vector
  IndexedVec,
}

impl StorageMetadata {
  pub fn choices_string() -> String {
    vec![Self::Sorted, Self::IndexedVec]
      .into_iter()
      .map(|m| format!("{:?}", m.to_string()))
      .collect::<Vec<_>>()
      .join(", ")
  }
}

impl std::fmt::Display for StorageMetadata {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(match self {
      Self::Sorted => "sorted",
      Self::IndexedVec => "indexed_vec",
    })
  }
}

impl FromStr for StorageMetadata {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "sorted" => Ok(Self::Sorted),
      "indexed_vec" => Ok(Self::IndexedVec),
      _ => Err(format!("Unknown storage type `{}`", s)),
    }
  }
}

impl StorageMetadata {
  /// Use the metadata to create an empty storage
  pub fn create_empty_storage<Prov: Provenance>(&self) -> DynamicCollection<Prov> {
    match self {
      Self::Sorted => DynamicCollection::Sorted(DynamicSortedCollection::empty()),
      Self::IndexedVec => DynamicCollection::IndexedVec(DynamicIndexedVecCollection::empty()),
    }
  }

  /// Use the metadata to create a storage with the elements in a given vector
  pub fn from_vec<Prov: Provenance>(&self, vec: Vec<DynamicElement<Prov>>, ctx: &Prov) -> DynamicCollection<Prov> {
    match self {
      Self::Sorted => DynamicCollection::Sorted(DynamicSortedCollection::from_vec(vec, ctx)),
      Self::IndexedVec => DynamicCollection::IndexedVec(DynamicIndexedVecCollection::from_vec(vec, ctx)),
    }
  }
}
