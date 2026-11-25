use std::cell::Cell;

use scallop_core::common::tuple::Tuple;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

use super::CountedBatch;

#[test]
fn test_dynamic_product_1() {
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
      &DynamicDataflow::dynamic_relation(&source_1).product(DynamicDataflow::dynamic_relation(&source_2), &ctx),
      &rt,
    )
  }

  expect_collection(
    &target.complete(&ctx).into(),
    vec![
      ((0i8, 1i8), (0i8, 2i8)),
      ((0i8, 1i8), (1i8, 5i8)),
      ((1i8, 2i8), (0i8, 2i8)),
      ((1i8, 2i8), (1i8, 5i8)),
    ],
  );
}

#[test]
fn test_dynamic_product_reuses_contiguous_rows_and_preserves_streaming_order() {
  let ctx = natural::NaturalProvenance;
  let left = vec![DynamicElement::new(10i32, 2), DynamicElement::new(11i32, 3)];
  let right = [DynamicElement::new(20i32, 5), DynamicElement::new(21i32, 7)];
  for contiguous in [true, false] {
    let reads = Cell::new(0);
    let right_batch = CountedBatch::new(&right, &reads);
    let right_batch = if contiguous {
      right_batch.contiguous()
    } else {
      right_batch
    };
    let product = DynamicDataflow::vec(&left).product(DynamicDataflow::new(right_batch), &ctx);
    let mut batch = product.iter_recent().next().unwrap();
    let first = batch.next().unwrap();
    assert_eq!((first.tuple, first.tag), (Tuple::from((10i32, 20i32)), 10));
    let snapshot = batch.clone();
    let remaining = [
      (Tuple::from((10i32, 21i32)), 14),
      (Tuple::from((11i32, 20i32)), 15),
      (Tuple::from((11i32, 21i32)), 21),
    ];
    assert!(batch.map(|row| (row.tuple, row.tag)).eq(remaining.iter().cloned()));
    assert_eq!(reads.get(), if contiguous { 0 } else { left.len() * right.len() });
    assert!(snapshot.map(|row| (row.tuple, row.tag)).eq(remaining));
  }
}

#[test]
fn test_dynamic_product_with_known_empty_right_does_not_read_left() {
  let ctx = unit::UnitProvenance;
  let left = [
    DynamicElement::new(10i32, ctx.one()),
    DynamicElement::new(11i32, ctx.one()),
  ];
  let right = Vec::new();
  let reads = Cell::new(0);
  let product = DynamicDataflow::new(CountedBatch::new(&left, &reads)).product(DynamicDataflow::vec(&right), &ctx);
  assert!(product.iter_recent().flatten().next().is_none());
  assert_eq!(reads.get(), 0);
}

#[test]
fn test_dynamic_product_recent_keeps_stable_recent_pairing_order() {
  let ctx = unit::UnitProvenance;
  let relation = |stable: i32, recent: i32| {
    let relation = DynamicRelation::new();
    let rows = |value| DynamicSortedCollection::from_vec_unchecked(vec![DynamicElement::new(value, ctx.one())]);
    relation.stable.borrow_mut().push(rows(stable));
    *relation.recent.borrow_mut() = rows(recent);
    relation
  };
  let left = relation(10, 11);
  let right = relation(20, 21);
  let product = DynamicDataflow::dynamic_relation(&left).product(DynamicDataflow::dynamic_relation(&right), &ctx);
  assert!(product
    .iter_recent()
    .flatten()
    .map(|row| row.tuple)
    .eq([(10i32, 21i32), (11, 20), (11, 21)].map(Tuple::from)));
}
