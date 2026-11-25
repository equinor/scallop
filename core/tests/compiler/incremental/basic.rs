use scallop_core::common::input_tag::DynamicInputTag;
use scallop_core::common::output_option::{OutputCSVFile, OutputFile, OutputOption};
use scallop_core::common::tuple::Tuple;
use scallop_core::compiler::front::FrontContext;
use scallop_core::compiler::{compile_string_to_ram, ram};

#[test]
fn front_context_compilation_is_isolated_and_transactional() {
  let mut original = FrontContext::new();
  original.compile_string("rel base = {1}".into()).unwrap();

  let mut fork = original.clone();
  fork.compile_string("rel added(x) = base(x)".into()).unwrap();
  assert!(fork.type_inference().has_relation("added"));
  assert!(!original.type_inference().has_relation("added"));

  let invalid = "type pending(x: i32)\nrel pending = {\"not an integer\"}";
  assert!(fork.compile_string(invalid.into()).is_err());
  assert!(!fork.type_inference().has_relation("pending"));
  fork.compile_string("rel pending(x) = added(x)".into()).unwrap();
}

#[test]
fn incr_1() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(0, x)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.contains("edge"));
  assert!(incr.contains("path"));
}

#[test]
fn incr_2() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2), (2, 3)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.is_empty());
}

#[test]
fn incr_3() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ edge(a, b) /\ path(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.len() == 1);
  assert!(incr.contains("edge"));
}

#[test]
fn incr_4() {
  let r1 = compile_string_to_ram(
    r#"
    rel R = {(0, 1), (1, 2)}
    rel Q(0, b) = R(0, b)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel R = {(0, 1), (1, 2)}
    rel Q(1, b) = R(1, b)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.len() == 1);
  assert!(incr.contains("R"));
}

#[test]
fn incr_5() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(1, _)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.contains("edge"));
  assert!(incr.contains("path"));
}

#[test]
fn ram_reuse_invalidates_dependents_of_one_changed_input() {
  let program = compile_string_to_ram(
    r"
    rel left = {1}
    rel right = {1}
    rel joined(x) = left(x) /\ right(x)
    rel result(x) = joined(x)
    query result
    "
    .to_owned(),
  )
  .unwrap();

  let reusable = program.persistent_relations_with_need_update_relations(&program, &["left".to_owned()].into());
  assert!(reusable.contains("right"));
  assert!(!reusable.contains("left"));
  assert!(!reusable.contains("joined"));
  assert!(!reusable.contains("result"));
}

#[test]
fn ram_reuse_invalidates_a_cycle_when_an_internal_update_changes() {
  let original = compile_string_to_ram(
    r"
    rel seed = {1}
    rel a(x) = seed(x) \/ b(x)
    rel b(x) = a(x)
    rel result(x) = b(x)
    query result
    "
    .to_owned(),
  )
  .unwrap();
  let mut updated = original.clone();
  let stratum = updated.relation_to_stratum["b"];
  let update = updated.strata[stratum]
    .updates
    .iter_mut()
    .find(|update| update.target == "b" && update.dependency().contains("a"))
    .unwrap();
  update.dataflow = ram::Dataflow::Find(Box::new(update.dataflow.clone()), Tuple::from((2_i32,)));

  let reusable = original.persistent_relations(&updated);
  assert!(reusable.contains("seed"));
  assert!(!reusable.contains("a"));
  assert!(!reusable.contains("b"));
  assert!(!reusable.contains("result"));
}

#[test]
fn ram_reuse_invalidates_updates_without_dependencies() {
  let original = compile_string_to_ram("rel seed = {1}\nrel result(x) = seed(x)\nquery result".to_owned()).unwrap();
  let mut updated = original.clone();
  let stratum = updated.relation_to_stratum["result"];
  updated.strata[stratum].updates.push(ram::Update {
    target: "result".to_owned(),
    dataflow: ram::Dataflow::UntaggedVec(vec![Tuple::from((2_i32,))]),
  });

  let reusable = original.persistent_relations(&updated);
  assert!(reusable.contains("seed"));
  assert!(!reusable.contains("result"));
}

#[test]
fn ram_facts_order_nan_tags_by_exclusion_id() {
  let fact = ram::Fact {
    tag: DynamicInputTag::ExclusiveFloat(f64::NAN, 1),
    tuple: Tuple::unit(),
  };
  let other_tag = ram::Fact {
    tag: DynamicInputTag::ExclusiveFloat(-f64::NAN, 2),
    ..fact.clone()
  };
  assert_eq!(fact, fact.clone());
  assert!(fact < other_tag);
}

#[test]
fn ram_output_metadata_changes_order_without_invalidating_reuse() {
  let original = compile_string_to_ram("rel a = {1}".to_string()).unwrap();
  let mut updated = original.clone();
  let stratum = updated.relation_to_stratum["a"];
  let output_relation = updated.strata[stratum].relations.get_mut("a").unwrap();
  output_relation.output = OutputOption::File(OutputFile::CSV(OutputCSVFile::new("a.csv".into())));
  assert!(original.relation_unchecked("a") < output_relation);
  assert!(original.persistent_relations(&updated).contains("a"));
  assert!(original
    .persistent_relations_with_need_update_relations(&updated, &["a".to_string()].into())
    .is_empty());
}
