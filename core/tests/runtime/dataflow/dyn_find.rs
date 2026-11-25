use scallop_core::common::tuple::Tuple;
use scallop_core::runtime::dynamic::dataflow::batching::Batch;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

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
