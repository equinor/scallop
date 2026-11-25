use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use scallop_core::common::expr::*;
use scallop_core::common::foreign_tensor::{DynamicExternalTensor, ExternalTensor, TensorShape};
use scallop_core::common::tuple::Tuple;
use scallop_core::common::value::Value;
use scallop_core::runtime::dynamic::dataflow::{Dataflow, DynamicDataflow};
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[test]
fn test_dynamic_exclusion_1() {
  let ctx = proofs::ProofsProvenance::<RcFamily>::default();
  let rt = RuntimeEnvironment::default();

  // Relations
  let mut source = DynamicRelation::<proofs::ProofsProvenance<RcFamily>>::new();
  let mut target = DynamicRelation::<proofs::ProofsProvenance<RcFamily>>::new();
  source.insert_untagged(&ctx, vec![(0,), (1,)]);

  // Untagged vec for exclusion
  let exc = vec![("red".to_string(),).into(), ("blue".to_string(),).into()];

  // Iterate until fixpoint
  let mut first_time = true;
  while source.changed(&ctx, rt.get_default_scheduler())
    || target.changed(&ctx, rt.get_default_scheduler())
    || first_time
  {
    target.insert_dataflow_recent(
      &ctx,
      &dataflow::DynamicDataflow::dynamic_relation(&source)
        .dynamic_exclusion(dataflow::DynamicDataflow::untagged_vec(&ctx, exc.clone()), &ctx, &rt)
        .project((Expr::access((0, 0)), Expr::access((1, 0))).into(), &rt),
      &rt,
    );
    first_time = false;
  }

  // Inspect the result
  expect_collection(
    &target.complete(&ctx).into(),
    vec![
      (0, "red".to_string()),
      (0, "blue".to_string()),
      (1, "red".to_string()),
      (1, "blue".to_string()),
    ],
  );
}

#[test]
fn test_exclusion_replay_keeps_row_order_and_groups() {
  let ctx = proofs::ProofsProvenance::<RcFamily>::default();
  let rt = RuntimeEnvironment::default();
  let left = vec![
    DynamicElement::new(1i32, ctx.one()),
    DynamicElement::new(2i32, ctx.one()),
  ];
  let choices = DynamicDataflow::untagged_vec(&ctx, vec!["red".into(), "blue".into()]);
  let source = DynamicDataflow::vec(&left).dynamic_exclusion(choices, &ctx, &rt);

  let rows: Vec<_> = source.iter_recent().flatten().collect();
  assert!(rows
    .iter()
    .map(|row| row.tuple.clone())
    .eq([(1i32, "red"), (1, "blue"), (2, "red"), (2, "blue")].map(Tuple::from)));
  assert!(ctx.discard(&ctx.mult(&rows[0].tag, &rows[1].tag)));
  assert!(ctx.discard(&ctx.mult(&rows[2].tag, &rows[3].tag)));
  assert!(!ctx.discard(&ctx.mult(&rows[0].tag, &rows[2].tag)));

  let mut replay = source.iter_recent().flatten();
  let first = replay.next().unwrap();
  assert_eq!(first.tuple, rows[0].tuple);
  assert!(ctx.discard(&ctx.mult(&first.tag, &rows[1].tag)));
}

#[test]
fn test_untagged_replay_clones_only_rows_it_reads() {
  struct CountedTensor(Arc<AtomicUsize>);

  impl Clone for CountedTensor {
    fn clone(&self) -> Self {
      self.0.fetch_add(1, Ordering::Relaxed);
      Self(Arc::clone(&self.0))
    }
  }

  impl ExternalTensor for CountedTensor {
    fn shape(&self) -> TensorShape {
      TensorShape::scalar()
    }

    fn get_f64(&self) -> f64 {
      0.0
    }
  }

  let copies = Arc::new(AtomicUsize::new(0));
  let tuples = (0..3i32)
    .map(|i| {
      let tensor = Value::Tensor(DynamicExternalTensor::new(CountedTensor(Arc::clone(&copies))));
      Tuple::from((i, tensor))
    })
    .collect();
  let df = DynamicDataflow::untagged_vec(&unit::UnitProvenance, tuples);
  let mut batch = df.iter_recent().next().unwrap();
  let mut replay = batch.clone();
  assert_eq!(copies.load(Ordering::Relaxed), 0);

  assert_eq!(batch.next().unwrap().tuple[0].as_i32(), 0);
  assert_eq!(batch.next().unwrap().tuple[0].as_i32(), 1);
  assert_eq!(replay.next().unwrap().tuple[0].as_i32(), 0);
  assert_eq!(copies.load(Ordering::Relaxed), 3);
}
