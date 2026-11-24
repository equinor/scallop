use std::cell::RefCell;
use std::rc::Rc;

use scallop_core::compiler::compile_string_to_ram;
use scallop_core::runtime::dynamic;
use scallop_core::runtime::env;
use scallop_core::runtime::monitor::{Monitor, MonitorRegistry};
use scallop_core::runtime::provenance::*;
use scallop_core::utils::RcFamily;

#[test]
fn incr_exec_1() {
  let ctx = unit::UnitProvenance;
  let runtime = env::RuntimeEnvironment::default();
  let mut exec_ctx = dynamic::DynamicExecutionContext::<_, RcFamily>::new();

  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();

  exec_ctx.incremental_execute(r1, &runtime, &ctx).expect("Runtime Error");

  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(0, x)
  "#
    .to_string(),
  )
  .unwrap();

  exec_ctx.incremental_execute(r2, &runtime, &ctx).expect("Runtime Error");
}

#[test]
fn monitored_incremental_execution_reports_strata_in_order() {
  #[derive(Clone, Default)]
  struct StrataMonitor(Rc<RefCell<Vec<usize>>>);

  impl Monitor<unit::UnitProvenance> for StrataMonitor {
    fn name(&self) -> &'static str {
      "strata"
    }

    fn observe_executing_stratum(&self, stratum_id: usize) {
      self.0.borrow_mut().push(stratum_id);
    }
  }

  let program = compile_string_to_ram(
    "rel candidates = {1, 2} rel excluded = {2} rel selected(x) = candidates(x) and not excluded(x) query selected"
      .to_owned(),
  )
  .unwrap();
  let stratum_count = program.strata.len();
  assert!(stratum_count > 1);
  let monitor = StrataMonitor::default();
  let mut registry = MonitorRegistry::new();
  registry.register(monitor.clone());

  let mut exec = dynamic::DynamicExecutionContext::<_, RcFamily>::new();
  exec
    .incremental_execute_with_monitor(
      program,
      &env::RuntimeEnvironment::default(),
      &unit::UnitProvenance,
      &registry.load_monitors(&["strata"]),
    )
    .unwrap();

  assert!(exec.idb.has_relation("selected"));
  assert_eq!(*monitor.0.borrow(), (0..stratum_count).collect::<Vec<_>>());
}

#[test]
fn monitored_execution_derives_facts_for_new_entities() {
  let mut exec = scallop_core::integrate::InterpretContext::<_, RcFamily>::new(
    r#"
    type Expr = Const(i32) | Add(Expr, Expr)
    rel expr($parse_entity("Const(42)"))
    rel result(v) = expr(e) and case e is Const(v)
    query result
    "#
    .to_owned(),
    unit::UnitProvenance,
  )
  .unwrap();
  exec.run_with_monitor(&()).unwrap();

  let idb = exec.idb();
  let result = idb.get_output_collection_ref("result").unwrap();
  assert_eq!(
    result.iter().map(|(_, tuple)| tuple[0].as_i32()).collect::<Vec<_>>(),
    [42]
  );
}
