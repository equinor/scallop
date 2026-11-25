use std::cell::RefCell;

use scallop_core::common::expr::Expr;
use scallop_core::common::tuple::Tuple;
use scallop_core::compiler::ram;
use scallop_core::runtime::database::StorageMetadata;
use scallop_core::runtime::dynamic::dataflow::batching::{Batch, DynamicBatches, ElementsBatch};
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn simple_relation_dataflow() {
  let ctx = unit::UnitProvenance;
  let rt = RuntimeEnvironment::default();
  let sche = Scheduler::LFP;

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&ctx, vec![(0usize, 1usize), (1usize, 2usize)]);

  // Iterate until fixpoint
  while source.changed(&ctx, &sche) || target.changed(&ctx, &sche) {
    target.insert_dataflow_recent(&ctx, &DynamicDataflow::dynamic_relation(&source), &rt);
  }

  expect_collection(&target.complete(&ctx).into(), vec![(0usize, 1usize), (1usize, 2usize)]);
}

#[test]
fn sorted_insertion_keeps_duplicate_tag_order_and_sorts_unknown_batches() {
  let ctx = add_mult_prob::AddMultProbProvenance::default();
  let rt = RuntimeEnvironment::default();
  let rows = vec![
    DynamicElement::new(2i32, 0.5),
    DynamicElement::new(1i32, 0.25),
    DynamicElement::new(2i32, f64::EPSILON / 4.0),
    DynamicElement::new(2i32, f64::EPSILON / 4.0),
  ];
  for explicitly_sorted in [true, false] {
    let mut target = DynamicRelation::new();
    let source = DynamicDataflow::vec(&rows);
    let source = if explicitly_sorted { source.sorted() } else { source };
    let source = source.filter(Expr::constant(true), &rt);
    assert_eq!(
      source.iter_recent().next().unwrap().is_tuple_sorted(),
      explicitly_sorted
    );
    target.insert_dataflow_recent(&ctx, &source, &rt);
    drop(source);
    assert!(target.changed(&ctx, &Scheduler::LFP));
    assert!(target
      .recent
      .borrow()
      .elements
      .iter()
      .map(|row| (row.tuple.as_i32(), row.tag))
      .eq([(1, 0.25), (2, 0.5)]));
  }
}

#[test]
fn stable_and_recent_cursors_keep_their_own_position_and_borrow() {
  let source = DynamicRelation::<unit::UnitProvenance>::new();
  let rows = |key| {
    DynamicSortedCollection::from_vec_unchecked(
      [(key, 0), (key, 1), (key + 1, 0), (key + 2, 0)]
        .map(|tuple| DynamicElement::new(tuple, unit::Unit))
        .into(),
    )
  };
  source.stable.borrow_mut().extend([rows(0i32), rows(10)]);
  *source.recent.borrow_mut() = rows(20);

  let df = DynamicDataflow::dynamic_relation(&source);
  let mut stable = df.iter_stable();
  assert_eq!(stable.size_hint(), (2, Some(2)));
  assert_eq!(stable.next().unwrap().next().unwrap().tuple, Tuple::from((0i32, 0i32)));
  let stable_batch = stable.next().unwrap();
  assert_eq!(stable.size_hint(), (0, Some(0)));
  drop(stable);
  let recent_batch = df.iter_recent().next().unwrap();
  assert!(source.stable.try_borrow_mut().is_err());
  assert!(source.recent.try_borrow_mut().is_err());

  for (mut batch, key) in [(stable_batch, 10i32), (recent_batch, 20i32)] {
    assert!(batch.is_tuple_sorted());
    let row = batch.search_until(&(key, 1i32).into()).unwrap();
    assert_eq!(row.tuple, Tuple::from((key, 1i32)));
    assert_eq!(batch.remaining_slice().unwrap().len(), 2);
    let mut replay = batch.clone();
    let row = batch.search_elem_0_until(&(key + 2).into()).unwrap();
    assert_eq!(row.tuple, Tuple::from((key + 2, 0i32)));
    assert!(batch.next().is_none());
    replay.step(1);
    assert_eq!(replay.next().unwrap().tuple, row.tuple);
  }
  assert!(source.stable.try_borrow_mut().is_ok());
  assert!(source.recent.try_borrow_mut().is_ok());
}

#[test]
fn completed_outputs_keep_sorted_rows_or_convert_to_indexed_storage() {
  let ctx = natural::NaturalProvenance;
  let rt = RuntimeEnvironment::default();
  let mut iteration = DynamicIteration::new();
  iteration.create_dynamic_relation("source");
  iteration.create_dynamic_relation("sorted");
  iteration.create_dynamic_relation("indexed");
  iteration.get_dynamic_relation_unsafe("source").insert_tagged(
    &ctx,
    vec![(Some(2), (2usize, 9i32)), (Some(3), (0, 7)), (Some(5), (2, 9))],
  );
  for (target, storage) in [
    ("sorted", StorageMetadata::Sorted),
    ("indexed", StorageMetadata::IndexedVec),
  ] {
    iteration.add_update_dataflow(target, ram::Dataflow::relation("source"));
    iteration.add_output_relation(target, &storage);
  }
  let outputs = iteration.run(&ctx, &rt);
  let expected = [(Tuple::from((0usize, 7i32)), 3), (Tuple::from((2usize, 9i32)), 7)];
  assert!(matches!(outputs["sorted"], DynamicCollection::Sorted(_)));
  assert!(matches!(outputs["indexed"], DynamicCollection::IndexedVec(_)));
  for relation in outputs.values() {
    assert!(relation
      .iter()
      .map(|row| (row.tuple, row.tag))
      .eq(expected.iter().cloned()));
  }
}

#[test]
fn materialized_insertion_preserves_the_allocation_while_discarding_and_consolidating() {
  type Prov = natural::NaturalProvenance;

  #[derive(Clone)]
  struct OwnedRows<'a>(&'a RefCell<Option<DynamicElements<Prov>>>);

  impl<'a> Dataflow<'a, Prov> for OwnedRows<'a> {
    fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
      DynamicBatches::single(ElementsBatch::new(self.0.borrow_mut().take().unwrap()))
    }
  }

  let ctx = natural::NaturalProvenance;
  let rt = RuntimeEnvironment {
    early_discard: true,
    ..RuntimeEnvironment::default()
  };
  let rows = vec![
    DynamicElement::new(3i32, 0),
    DynamicElement::new(2i32, 5),
    DynamicElement::new(1i32, 2),
    DynamicElement::new(2i32, 7),
  ];
  let allocation = rows.as_ptr();
  let rows = RefCell::new(Some(rows));
  let mut target = DynamicRelation::new();
  target.insert_dataflow_recent(&ctx, &DynamicDataflow::new(OwnedRows(&rows)), &rt);
  assert!(target.changed(&ctx, &Scheduler::LFP));
  let recent = target.recent.borrow();
  assert_eq!(recent.elements.as_ptr(), allocation);
  assert!(recent
    .elements
    .iter()
    .map(|row| (row.tuple.as_i32(), row.tag))
    .eq([(1, 2), (2, 12)]));
}
