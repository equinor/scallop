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
