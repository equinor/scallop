use scallop_core::common::foreign_aggregates::aggregate_top_k_helper;
use scallop_core::integrate::*;

#[test]
fn test_uniform_sample() {
  let result = interpret_string(
    r#"
    rel numbers = {0, 1, 2, 3}
    rel sampled_number(x) = x := uniform<1>(x: numbers(x))
  "#
    .to_string(),
  )
  .expect("Failed executing");
  let sampled_number = result
    .get_output_collection("sampled_number")
    .expect("Cannot get `sampled_number` relation");
  assert_eq!(sampled_number.len(), 1, "There should be only one number being sampled");
  let number = sampled_number.ith_tuple(0).expect("There should be one number")[0].as_i32();
  assert!((0..4).contains(&number), "number should be anything between 0 and 3");
}

#[test]
fn test_top_k_prefers_numeric_weights_to_nan() {
  let weights = [f64::NAN, 0.4, 0.7, -f64::NAN, 0.7];
  let mut selected = aggregate_top_k_helper(weights.len(), 3, |i| weights[i]);
  selected.sort_unstable();
  assert_eq!(selected, [1, 2, 4]);
  assert_eq!(aggregate_top_k_helper(weights.len(), 1, |i| weights[i]), [2]);
}
