use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use scallop_core::common::expr::*;
use scallop_core::common::foreign_predicate::ForeignPredicate;
use scallop_core::common::input_tag::DynamicInputTag;
use scallop_core::common::tuple::*;
use scallop_core::common::value::*;
use scallop_core::common::value_type::ValueType;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;

#[test]
fn test_dyn_dataflow_free_range() {
  let runtime = RuntimeEnvironment::new_std();
  let ctx = unit::UnitProvenance::new();
  let df = DynamicDataflow::foreign_predicate_ground(
    "range#usize".to_string(),
    vec![Value::USize(1), Value::USize(5)],
    true,
    &ctx,
    &runtime,
  );
  let batch = df.iter_recent().next().unwrap().collect::<Vec<_>>();
  for i in 1..5 {
    assert_eq!(batch[i - 1].tuple, Tuple::from(vec![Value::USize(i)]));
  }
}

#[test]
fn test_dyn_dataflow_soft_lt_1() {
  let runtime = RuntimeEnvironment::new_std();
  let ctx = min_max_prob::MinMaxProbProvenance::new();
  let source_df = vec![
    DynamicElement::new((1.0, -1.0), 1.0),
    DynamicElement::new((1.0, 1.0), 1.0),
    DynamicElement::new((1.0, 5.0), 1.0),
  ];
  let df = DynamicDataflow::vec(&source_df).foreign_predicate_constraint(
    "soft_lt#f64".to_string(),
    vec![Expr::access(0), Expr::access(1)],
    &ctx,
    &runtime,
  );
  let batch = df.iter_recent().next().unwrap().collect::<Vec<_>>();
  for elem in batch {
    let tup: (f64, f64) = elem.tuple.as_tuple();
    if tup.0 < tup.1 {
      assert!(elem.tag > 0.5);
    } else if tup.0 == tup.1 {
      assert!(elem.tag == 0.5);
    } else {
      assert!(elem.tag < 0.5);
    }
  }
}

#[test]
fn test_dyn_dataflow_join_range() {
  let runtime = RuntimeEnvironment::new_std();
  let ctx = unit::UnitProvenance::new();
  let data = vec![
    DynamicElement::new((1usize, 3usize), unit::Unit),
    DynamicElement::new((10usize, 10usize), unit::Unit),
    DynamicElement::new((100usize, 101usize), unit::Unit),
  ];
  let df = DynamicDataflow::vec(&data).foreign_predicate_join(
    "range#usize".to_string(),
    vec![Expr::access(0), Expr::access(1)],
    &ctx,
    &runtime,
  );
  let batch = df.iter_recent().next().unwrap().collect::<Vec<_>>();
  assert_eq!(
    AsTuple::<((usize, usize), (usize,))>::as_tuple(&batch[0].tuple),
    ((1usize, 3usize), (1usize,))
  );
  assert_eq!(
    AsTuple::<((usize, usize), (usize,))>::as_tuple(&batch[1].tuple),
    ((1usize, 3usize), (2usize,))
  );
  assert_eq!(
    AsTuple::<((usize, usize), (usize,))>::as_tuple(&batch[2].tuple),
    ((100usize, 101usize), (100usize,))
  );
}

struct StatefulPredicate {
  num_free: usize,
  calls: Arc<AtomicUsize>,
  local_calls: AtomicUsize,
}

impl Clone for StatefulPredicate {
  fn clone(&self) -> Self {
    Self {
      num_free: self.num_free,
      calls: Arc::clone(&self.calls),
      local_calls: AtomicUsize::new(self.local_calls.load(Ordering::Relaxed)),
    }
  }
}

impl ForeignPredicate for StatefulPredicate {
  fn name(&self) -> String {
    "stateful".to_string()
  }

  fn arity(&self) -> usize {
    1 + self.num_free
  }

  fn argument_type(&self, _: usize) -> ValueType {
    ValueType::I32
  }

  fn num_bounded(&self) -> usize {
    1
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    self.calls.fetch_add(1, Ordering::Relaxed);
    let tag = DynamicInputTag::Natural(self.local_calls.fetch_add(1, Ordering::Relaxed) + 1);
    if self.num_free > 0 {
      vec![(tag.clone(), vec![bounded[0].clone()]), (tag, vec![Value::I32(0)])]
    } else if bounded[0] == Value::I32(0) {
      vec![]
    } else {
      vec![(tag, vec![])]
    }
  }
}

fn stateful_predicate(num_free: usize) -> (RuntimeEnvironment, Arc<AtomicUsize>) {
  let mut runtime = RuntimeEnvironment::default();
  let calls = Arc::new(AtomicUsize::new(0));
  runtime
    .predicate_registry
    .register(StatefulPredicate {
      num_free,
      calls: Arc::clone(&calls),
      local_calls: AtomicUsize::new(0),
    })
    .unwrap();
  (runtime, calls)
}

#[test]
fn test_ground_foreign_predicate_materializes_on_active_phase_entry() {
  let (runtime, calls) = stateful_predicate(1);
  let ctx = natural::NaturalProvenance;
  for first_iteration in [true, false] {
    let df = DynamicDataflow::foreign_predicate_ground(
      "stateful".to_string(),
      vec![Value::I32(7)],
      first_iteration,
      &ctx,
      &runtime,
    );
    let inactive = if first_iteration {
      df.iter_stable()
    } else {
      df.iter_recent()
    };
    let before = usize::from(!first_iteration);
    assert_eq!(inactive.size_hint(), (0, Some(0)));
    assert_eq!(calls.load(Ordering::Relaxed), before);

    let mut active = if first_iteration {
      df.iter_recent()
    } else {
      df.iter_stable()
    };
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
    assert!(active
      .next()
      .unwrap()
      .map(|row| (row.tuple, row.tag))
      .eq([(Tuple::from((7i32,)), before + 1), (Tuple::from((0i32,)), before + 1),]));
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
  }
}

#[test]
fn test_foreign_constraint_evaluates_only_on_row_pull() {
  let (runtime, calls) = stateful_predicate(0);
  let ctx = natural::NaturalProvenance;
  let rows = vec![DynamicElement::new((0i32,), 3), DynamicElement::new((7i32,), 5)];
  let df = DynamicDataflow::vec(&rows).foreign_predicate_constraint(
    "stateful".to_string(),
    vec![Expr::access(0)],
    &ctx,
    &runtime,
  );
  let mut batches = df.iter_recent();
  assert_eq!(batches.size_hint(), (1, Some(1)));
  let mut batch = batches.next().unwrap();
  assert_eq!(calls.load(Ordering::Relaxed), 0);
  let row = batch.next().unwrap();
  assert_eq!((row.tuple, row.tag), (Tuple::from((7i32,)), 10));
  assert_eq!(calls.load(Ordering::Relaxed), 2);
  assert!(batch.next().is_none());
}

#[test]
fn test_foreign_join_prefetches_before_cloning_and_replays_independently() {
  let (runtime, calls) = stateful_predicate(1);
  let ctx = natural::NaturalProvenance;
  let rows = vec![DynamicElement::new((7i32,), 1), DynamicElement::new((9i32,), 1)];
  let df =
    DynamicDataflow::vec(&rows).foreign_predicate_join("stateful".to_string(), vec![Expr::access(0)], &ctx, &runtime);
  let mut batches = df.iter_recent();
  assert_eq!(batches.size_hint(), (1, Some(1)));
  assert_eq!(calls.load(Ordering::Relaxed), 0);
  let mut batch = batches.next().unwrap();
  assert_eq!(calls.load(Ordering::Relaxed), 1);
  let row = batch.next().unwrap();
  assert_eq!((row.tuple, row.tag), (Tuple::from(((7i32,), (7i32,))), 1));
  assert_eq!(calls.load(Ordering::Relaxed), 1);

  let replay = batch.clone();
  let expected = [
    (((7i32,), (0i32,)), 1),
    (((9i32,), (9i32,)), 2),
    (((9i32,), (0i32,)), 2),
  ]
  .map(|(tuple, tag)| (Tuple::from(tuple), tag));
  for cursor in [batch, replay] {
    assert!(cursor.map(|row| (row.tuple, row.tag)).eq(expected.iter().cloned()));
  }
  assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn test_foreign_output_moves_strings_and_registers_symbols_in_order() {
  #[derive(Clone)]
  struct Strings(Arc<AtomicUsize>);

  impl ForeignPredicate for Strings {
    fn name(&self) -> String {
      "strings".to_owned()
    }

    fn arity(&self) -> usize {
      3
    }

    fn argument_type(&self, index: usize) -> ValueType {
      if index == 1 {
        ValueType::String
      } else {
        ValueType::Symbol
      }
    }

    fn num_bounded(&self) -> usize {
      0
    }

    fn evaluate(&self, _: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
      let value = String::from("moved");
      self.0.store(value.as_ptr() as usize, Ordering::Relaxed);
      vec![(
        DynamicInputTag::Natural(3),
        vec![
          Value::SymbolString("first".to_owned()),
          Value::String(value),
          Value::SymbolString("second".to_owned()),
        ],
      )]
    }
  }

  let mut runtime = RuntimeEnvironment::default();
  let address = Arc::new(AtomicUsize::new(0));
  runtime
    .predicate_registry
    .register(Strings(Arc::clone(&address)))
    .unwrap();
  let df = DynamicDataflow::foreign_predicate_ground(
    "strings".to_owned(),
    vec![],
    true,
    &natural::NaturalProvenance,
    &runtime,
  );
  let row = df.iter_recent().flatten().next().unwrap();
  let first = runtime.symbol_registry.get_id("first").unwrap();
  let second = runtime.symbol_registry.get_id("second").unwrap();
  assert!(first < second);
  assert_eq!(row.tuple[0], Tuple::from(Value::Symbol(first)));
  assert_eq!(row.tuple[2], Tuple::from(Value::Symbol(second)));
  let Some(Value::String(value)) = row.tuple[1].get_value() else {
    panic!("expected an owned string");
  };
  assert_eq!(value, "moved");
  assert_eq!(value.as_ptr() as usize, address.load(Ordering::Relaxed));
  assert_eq!(row.tag, 3);
}
