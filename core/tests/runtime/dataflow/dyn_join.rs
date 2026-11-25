use std::cell::Cell;

use scallop_core::common::tuple::Tuple;
use scallop_core::runtime::dynamic::dataflow::batching::Batch;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

use super::CountedBatch;

#[test]
fn test_dynamic_join_1() {
  let ctx = unit::UnitProvenance;
  let rt = RuntimeEnvironment::new_std();

  // Relations
  let mut source_1 = DynamicRelation::<unit::UnitProvenance>::new();
  let mut source_2 = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source_1.insert_untagged(&ctx, vec![(0i8, 1i8), (1i8, 2i8)]);
  source_2.insert_untagged(&ctx, vec![(0i8, 2i8), (1i8, 5i8)]);

  // Iterate until fixpoint
  while source_1.changed(&ctx, rt.get_default_scheduler())
    || source_2.changed(&ctx, rt.get_default_scheduler())
    || target.changed(&ctx, rt.get_default_scheduler())
  {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::dynamic_relation(&source_1).join(DynamicDataflow::dynamic_relation(&source_2), &ctx),
      &rt,
    )
  }

  expect_collection(&target.complete(&ctx).into(), vec![(0i8, 1i8, 2i8), (1i8, 2i8, 5i8)]);
}

#[test]
fn test_dynamic_join_reads_matching_groups_once_in_left_major_order() {
  let ctx = natural::NaturalProvenance;
  let left = [
    DynamicElement::new((1i32, 10i32), 2),
    DynamicElement::new((1i32, 11i32), 3),
    DynamicElement::new((2i32, 12i32), 4),
    DynamicElement::new((3i32, 13i32), 5),
  ];
  let right = [
    DynamicElement::new((0i32, 19i32), 6),
    DynamicElement::new((1i32, 20i32), 7),
    DynamicElement::new((1i32, 21i32), 11),
    DynamicElement::new((3i32, 22i32), 13),
  ];
  let left_reads = Cell::new(0);
  let right_reads = Cell::new(0);
  let join = DynamicDataflow::new(CountedBatch::new(&left, &left_reads))
    .join(DynamicDataflow::new(CountedBatch::new(&right, &right_reads)), &ctx);
  let mut batch = join.iter_recent().next().unwrap();
  let first = batch.next().unwrap();
  assert_eq!((first.tuple, first.tag), (Tuple::from((1i32, 10i32, 20i32)), 14));
  assert_eq!(left_reads.get(), 1);

  let snapshot = batch.clone();
  let remaining = [
    (Tuple::from((1i32, 10i32, 21i32)), 22),
    (Tuple::from((1i32, 11i32, 20i32)), 21),
    (Tuple::from((1i32, 11i32, 21i32)), 33),
    (Tuple::from((3i32, 13i32, 22i32)), 65),
  ];
  assert!(batch.map(|e| (e.tuple, e.tag)).eq(remaining.iter().cloned()));
  assert_eq!(left_reads.get(), left.len());
  assert_eq!(right_reads.get(), right.len());
  assert!(snapshot.map(|e| (e.tuple, e.tag)).eq(remaining));
}

#[test]
fn test_dynamic_antijoin_reads_matching_left_rows_once() {
  let ctx = boolean::BooleanProvenance;
  let left = [(0i32, 10i32), (1, 11), (1, 12), (3, 13), (4, 14)].map(|pair| DynamicElement::new(pair, true));
  let right = vec![
    DynamicElement::new(1i32, true),
    DynamicElement::new(2i32, true),
    DynamicElement::new(3i32, false),
  ];
  let reads = Cell::new(0);
  let antijoin = DynamicDataflow::new(CountedBatch::new(&left, &reads)).antijoin(DynamicDataflow::vec(&right), &ctx);
  let expected = [true, false, false, true, true];
  assert!(antijoin
    .iter_recent()
    .flatten()
    .map(|e| (e.tuple, e.tag))
    .eq(left.iter().map(|e| e.tuple.clone()).zip(expected)));
  assert_eq!(reads.get(), left.len());

  let ctx = unit::UnitProvenance;
  let left = vec![
    DynamicElement::new((1i32, 11i32), ctx.one()),
    DynamicElement::new((1i32, 12i32), ctx.one()),
    DynamicElement::new((2i32, 13i32), ctx.one()),
  ];
  let right = vec![DynamicElement::new(1i32, ctx.one())];
  let antijoin = DynamicDataflow::vec(&left).antijoin(DynamicDataflow::vec(&right), &ctx);
  assert!(antijoin
    .iter_recent()
    .flatten()
    .map(|e| e.tuple)
    .eq([Tuple::from((2i32, 13i32))]));
}

#[test]
fn test_dynamic_batch_forwards_first_field_seek() {
  let ctx = unit::UnitProvenance;
  let mut relation = DynamicRelation::new();
  let rt = RuntimeEnvironment::new_std();
  relation.insert_untagged(&ctx, vec![(1i32, 10i32), (2, 20), (3, 30), (3, 31)]);
  assert!(relation.changed(&ctx, rt.get_default_scheduler()));
  let mut batch = DynamicDataflow::dynamic_relation(&relation)
    .iter_recent()
    .next()
    .unwrap();
  assert_eq!(batch.next().unwrap().tuple, Tuple::from((1i32, 10i32)));
  assert_eq!(
    batch.search_elem_0_until(&Tuple::from(3i32)).unwrap().tuple,
    Tuple::from((3i32, 30i32))
  );
  assert_eq!(batch.next().unwrap().tuple, Tuple::from((3i32, 31i32)));
}
