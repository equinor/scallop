use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn expression_tags_preserve_runtime_exclusions() {
  let context = top_k_proofs::TopKProofsProvenance::<scallop_core::utils::RcFamily>::new(8, false);
  let result = scallop_core::integrate::interpret_string_with_ctx(
    r#"
      rel variable = {1, 2}
      rel { assignment(x, true); assignment(x, false) } = variable(x)
      type denominator(i32, f32)
      rel denominator = {(1, 1.0), (2, 2.0)}
      rel (1.0 / p)::weighted(x, b) = assignment(x, b), denominator(x, p)
      rel impossible(x) = weighted(x, true), weighted(x, false)
    "#
    .into(),
    context,
  )
  .unwrap();
  expect_output_collection_with_tag(
    "weighted",
    result.get_output_collection_ref("weighted").unwrap(),
    vec![(1.0, (1, false)), (1.0, (1, true)), (0.5, (2, false)), (0.5, (2, true))],
    |expected, actual| expected == actual,
  );
  assert!(result.get_output_collection_ref("impossible").unwrap().is_empty());
}

#[test]
fn test_how_many_3_add_mult() {
  let ctx = add_mult_prob::AddMultProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel digit = {0.91::(0, 0), 0.01::(0, 1), 0.01::(0, 2), 0.01::(0, 3)}
      rel result(n) :- n = count(o: digit(o, 3))
    "#,
    ctx,
    ("result", vec![(0.99, (0usize,)), (0.01, (1usize,))]),
    add_mult_prob::AddMultProbProvenance::soft_cmp,
  )
}

#[test]
fn test_min_max_with_recursion() {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel enemy = {
        0.1::(1, 3), 0.1::(2, 3), 0.1::(3, 3),
        0.1::(1, 2), 0.8::(2, 2), 0.9::(3, 2),
        0.1::(1, 1), 0.1::(2, 1), 0.1::(3, 1),
      }
      rel edge = {
        (1, 1, 1, 2), (1, 1, 2, 1),
        (1, 2, 1, 3), (1, 2, 2, 2), (1, 2, 1, 1),
        (1, 3, 1, 2), (1, 3, 2, 3),
        (2, 1, 1, 1), (2, 1, 2, 2), (2, 1, 3, 1),
        (2, 2, 1, 2), (2, 2, 2, 3), (2, 2, 2, 1), (2, 2, 3, 2),
        (2, 3, 1, 3), (2, 3, 2, 2), (2, 3, 3, 3),
        (3, 1, 2, 1), (3, 1, 3, 2),
        (3, 2, 2, 2), (3, 2, 3, 1), (3, 2, 3, 3),
        (3, 3, 2, 3), (3, 3, 3, 2),
      }
      rel path(xa, ya, xb, yb) = edge(xa, ya, xb, yb), not enemy(xb, yb)
      rel path(xa, ya, xc, yc) = path(xa, ya, xb, yb), edge(xb, yb, xc, yc), not enemy(xc, yc)
      query path(3, 1, 3, 3)
    "#,
    ctx,
    ("path(3, 1, 3, 3)", vec![(0.9, (3, 1, 3, 3))]),
    min_max_prob::MinMaxProbProvenance::cmp,
  )
}

#[test]
fn test_discrete_count() {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel obj = {0.9::0, 0.5::1, 0.1::2}
      rel hard_count(n) = n := count!(id: obj(id))
      query hard_count
    "#,
    ctx,
    ("hard_count", vec![(1.0, (3usize,))]),
    min_max_prob::MinMaxProbProvenance::cmp,
  )
}
