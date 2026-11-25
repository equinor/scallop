use scallop_core::compiler::back::attributes::{GoalAttribute, StorageAttribute};
use scallop_core::compiler::back::{Attribute, Attributes};
use scallop_core::compiler::compile_string_to_ram;
use scallop_core::compiler::front::ast;
use scallop_core::compiler::front::attribute::*;
use scallop_core::compiler::front::FrontContext;
use scallop_core::runtime::database::StorageMetadata;
use scallop_core::testing::*;

mod attr_1 {
  use super::*;

  #[derive(Clone)]
  struct Foo;

  impl AttributeProcessor for Foo {
    fn name(&self) -> String {
      "foo".to_string()
    }

    fn apply(&self, _: &ast::Item, attr: &ast::Attribute) -> Result<AttributeAction, AttributeError> {
      if attr.num_pos_args() != 3 {
        Err(AttributeError::new_custom(
          "foo attribute requires 3 arguments".to_string(),
        ))
      } else {
        if attr.pos_arg(0).map(|arg| arg.is_tuple()) == Some(true) {
          Ok(AttributeAction::Nothing)
        } else {
          Err(AttributeError::new_custom(
            "foo attribute requires a tuple as the first argument".to_string(),
          ))
        }
      }
    }
  }

  #[test]
  fn unknown_attribute_1() {
    expect_front_compile_failure(
      r#"
      @foo((1, 2), 3, 4)
      type my_relation(a: i32, b: i32)
      "#,
      |s| s.contains("Unknown attribute `@foo`"),
    );
  }

  #[test]
  fn attr_1_test_2() {
    expect_front_compile_failure_with_modifier(
      r#"
      @foo((1, 2), 3)
      type my_relation(a: i32, b: i32)
      "#,
      |ctx: &mut FrontContext| {
        ctx
          .register_attribute_processor(Foo)
          .expect("Cannot register attribute");
      },
      |s| s.contains("foo attribute requires 3 arguments"),
    );
  }

  #[test]
  fn attr_1_test_3() {
    expect_front_compile_failure_with_modifier(
      r#"
      @foo("asdfasdf", 3, 5)
      type my_relation(a: i32, b: i32)
      "#,
      |ctx: &mut FrontContext| {
        ctx
          .register_attribute_processor(Foo)
          .expect("Cannot register attribute");
      },
      |s| s.contains("foo attribute requires a tuple as the first argument"),
    );
  }
}

#[test]
fn backend_attributes_keep_concrete_types_when_cloned() {
  let mut attrs = Attributes::new();
  attrs.insert(GoalAttribute);
  let attrs = attrs.clone();
  assert!(attrs.get::<GoalAttribute>().is_some());
  assert!(attrs.get::<StorageAttribute>().is_none());
  assert!(Attribute::new(GoalAttribute).get::<StorageAttribute>().is_none());
}

#[test]
fn storage_attribute_keeps_supported_modes() {
  let source = r#"
    @storage("indexed_vec")
    type indexed(i: usize, n: i32)
    rel indexed = {(0, 7), (2, 9)}
    @storage("sorted")
    rel result(n) = indexed(2, n)
    query result
  "#;
  let program = compile_string_to_ram(source.to_owned()).unwrap();
  assert_eq!(
    program.relation("indexed").unwrap().storage,
    StorageMetadata::IndexedVec
  );
  assert_eq!(program.relation("result").unwrap().storage, StorageMetadata::Sorted);
  expect_interpret_result(source, ("result", vec![(9i32,)]));

  expect_front_compile_failure(r#"@storage("dense_matrix") rel invalid = {(0, 1)}"#, |error| {
    error.contains("Choose from \"sorted\", \"indexed_vec\"")
  });
}

#[test]
fn indexed_join_uses_the_last_value_when_projecting_a_wide_relation() {
  expect_interpret_result(
    r#"
      @storage("indexed_vec")
      type indexed(i: usize, n: i32)
      rel indexed = {(0, 7), (0, 8), (2, 9)}
      type key(usize)
      rel key = {0, 1, 2}
      rel wide(i, n, i, n, i, n, i, n, i, n, i, n, i) = key(i), indexed(i, n)
      rel projected(i, n) = wide(_, _, _, _, _, _, _, _, _, _, _, n, i)
      query projected
    "#,
    ("projected", vec![(0usize, 8), (2, 9)]),
  );
}
