use scallop_core::common::input_tag::DynamicInputTag;
use scallop_core::common::output_option::{OutputCSVFile, OutputFile, OutputOption};
use scallop_core::common::tuple::Tuple;
use scallop_core::compiler::{compile_string_to_ram, ram};

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
