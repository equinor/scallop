use sdd::*;

#[test]
fn bool_formula_1() {
  let form = (bf(0) & bf(1) & bf(2)) | (bf(3) & bf(4));
  println!("{:?}", form);
}

#[test]
fn display_preserves_prefix_format() {
  let formula = !(bf(1) & bf_neg(20)) | (bf_true() & bf_false());
  assert_eq!(format!("{formula}"), "onap1n20atf");
  assert_eq!(formula.to_string(), "onap1n20atf");
}
