use std::cell::Cell;

use scallop_core::common::expr::Expr;
use scallop_core::common::tuple::Tuple;
use scallop_core::runtime::dynamic::dataflow::batching::Batch;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

use super::CountedBatch;

#[test]
fn test_dynamic_find_1() {
  let ctx = unit::UnitProvenance;
  let rt = RuntimeEnvironment::default();

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&ctx, vec![(0i8, 1i8), (1i8, 2i8)]);

  // Iterate until fixpoint
  while source.changed(&ctx, rt.get_default_scheduler()) || target.changed(&ctx, rt.get_default_scheduler()) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::find(DynamicDataflow::dynamic_relation(&source), 1i8.into()),
      &rt,
    )
  }

  expect_collection(&source.complete(&ctx).into(), vec![(0i8, 1i8), (1i8, 2i8)]);
  expect_collection(&target.complete(&ctx).into(), vec![(1i8, 2i8)]);
}

#[test]
fn test_sorted_collection_seeks_by_key_and_full_tuple() {
  let ctx = unit::UnitProvenance;
  let entries = (0..1024i32)
    .map(|key| DynamicElement::new((key, 0i32), ctx.one()))
    .chain([
      DynamicElement::new((1023i32, 1i32), ctx.one()),
      DynamicElement::new((1024i32, 0i32), ctx.one()),
    ])
    .collect();
  let rows = DynamicSortedCollection::<unit::UnitProvenance>::from_vec_unchecked(entries);
  let source = DynamicDataflow::dynamic_recent_sorted_collection(&rows);
  let mut batch = source.iter_recent().next().unwrap();
  assert_eq!(
    batch.search_elem_0_until(&1023i32.into()).unwrap().tuple,
    Tuple::from((1023i32, 0i32))
  );
  assert_eq!(
    batch.search_until(&(1023i32, 1i32).into()).unwrap().tuple,
    Tuple::from((1023i32, 1i32))
  );
  assert!(batch.search_elem_0_until(&1025i32.into()).is_none());
  assert!(batch.next().is_none());
  assert!(source
    .find(1023i32.into())
    .iter_recent()
    .flatten()
    .map(|row| row.tuple)
    .eq([(1023i32, 0i32), (1023, 1)].map(Tuple::from)));
}

#[test]
fn test_materialized_sort_seeks_without_replaying_and_keeps_equal_row_order() {
  let pulls = Cell::new(0);
  let rows = [
    ((9i32, 0i32), 1),
    ((3, 30), 2),
    ((3, 10), 3),
    ((7, 70), 4),
    ((3, 30), 5),
  ]
  .map(|(row, tag)| DynamicElement::<natural::NaturalProvenance>::new(row, tag));
  let source = DynamicDataflow::new(CountedBatch::new(&rows, &pulls)).sorted();
  let mut batch = source.iter_recent().next().unwrap();
  assert!(batch.remaining_slice().is_none());
  assert_eq!(pulls.get(), 0);
  let row = batch.search_until(&(3i32, 30i32).into()).unwrap();
  assert_eq!((row.tuple, row.tag), (Tuple::from((3i32, 30i32)), 2));
  assert_eq!(pulls.get(), rows.len());
  assert_eq!(batch.remaining_slice().unwrap().len(), 3);
  let row = batch.search_elem_0_until(&3i32.into()).unwrap();
  assert_eq!((row.tuple, row.tag), (Tuple::from((3i32, 30i32)), 5));
  assert_eq!(
    batch.search_elem_0_until(&8i32.into()).unwrap().tuple,
    Tuple::from((9i32, 0i32))
  );
  assert!(batch.next_elem().is_none());
  assert_eq!(pulls.get(), rows.len());
}

#[test]
fn test_sort_reuses_seekable_storage_but_buffers_sorted_streams() {
  let rt = RuntimeEnvironment::default();
  let rows = DynamicSortedCollection::<unit::UnitProvenance>::from_vec_unchecked(
    (0..16i32)
      .map(|key| DynamicElement::new((key, key + 1), unit::Unit))
      .collect(),
  );
  let source = DynamicDataflow::dynamic_recent_sorted_collection(&rows);
  let mut direct = source.clone().sorted().iter_recent().next().unwrap();
  assert_eq!(direct.remaining_slice().unwrap().as_ptr(), rows.elements.as_ptr());
  assert_eq!(
    direct.search_until(&(13i32, 14i32).into()).unwrap().tuple,
    Tuple::from((13i32, 14i32))
  );

  let filtered = source.filter(Expr::constant(true), &rt);
  assert!(filtered.iter_recent().next().unwrap().is_tuple_sorted());
  let mut buffered = filtered.sorted().iter_recent().next().unwrap();
  assert!(buffered.remaining_slice().is_none());
  assert_eq!(
    buffered.search_elem_0_until(&13i32.into()).unwrap().tuple,
    Tuple::from((13i32, 14i32))
  );
  assert_eq!(buffered.remaining_slice().unwrap().len(), 2);
}
