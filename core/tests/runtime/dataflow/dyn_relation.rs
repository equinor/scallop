use scallop_core::common::expr::Expr;
use scallop_core::runtime::dynamic::dataflow::batching::Batch;
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
