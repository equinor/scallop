use std::collections::BTreeSet;

use scallop_core::common::expr::*;
use scallop_core::common::foreign_aggregate::*;
use scallop_core::common::value_type::ValueType;
use scallop_core::compiler::ram::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::RcFamily;

#[test]
fn test_weighted_model_count_with_sparse_facts_and_negation() {
  let formula = DNFFormula::new(vec![
    Clause::new(vec![Literal::Pos(7), Literal::Neg(usize::MAX)]),
    Clause::new(vec![Literal::Pos(100), Literal::Pos(10_000)]),
  ]);
  let weight = |id: &usize| match *id {
    7 => 0.5,
    usize::MAX => 0.25,
    100 => 0.5,
    10_000 => 0.25,
    _ => panic!("unexpected fact ID {id}"),
  };
  // The two proofs are independent: 3/8 + 1/8 - (3/8 * 1/8).
  assert_eq!(formula.wmc(&RealSemiring, &weight), 29.0 / 64.0);
  let complement = CNFDNFFormula::cnf(vec![
    Clause::new(vec![Literal::Neg(7), Literal::Pos(usize::MAX)]),
    Clause::new(vec![Literal::Neg(100), Literal::Neg(10_000)]),
  ]);
  assert_eq!(complement.wmc(&RealSemiring, &weight), 35.0 / 64.0);
  assert_eq!(DNFFormula::zero().wmc(&RealSemiring, &weight), 0.0);
  assert_eq!(DNFFormula::one().wmc(&RealSemiring, &weight), 1.0);
}

#[test]
fn test_weighted_model_count_of_single_clauses() {
  let weight = |_: &usize| 0.25;
  let clause = Clause::new(vec![Literal::Pos(7), Literal::Neg(100)]);
  assert_eq!(
    DNFFormula::new(vec![clause.clone()]).wmc(&RealSemiring, &weight),
    0.1875
  );
  assert_eq!(CNFDNFFormula::cnf(vec![clause]).wmc(&RealSemiring, &weight), 0.8125);
  let repeated = Clause::new(vec![Literal::Pos(7), Literal::Pos(7)]);
  assert_eq!(DNFFormula::new(vec![repeated]).wmc(&RealSemiring, &weight), 0.25);
}

#[test]
fn test_weighted_model_count_through_rsdd_for_overlapping_proofs() {
  let formula = DNFFormula::new(
    (1..9)
      .map(|id| Clause::new(vec![Literal::Pos(0), Literal::Pos(id)]))
      .collect(),
  );
  let semiring = DualNumberSemiring::new(9);
  let count = formula.wmc(&semiring, &|id| semiring.singleton(0.5, *id));
  assert_eq!(count.real, 255.0 / 512.0);
  assert_eq!(count.deriv.indices(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert_eq!(
    count.deriv.data(),
    &[
      255.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0,
      1.0 / 256.0
    ]
  );
}

#[test]
fn test_disjunctions_preserve_first_seen_order_for_sparse_groups() {
  let mut disjunctions = Disjunctions::new();
  for (group, fact) in [(usize::MAX, 7), (0, 5), (usize::MAX, 3), (0, 11), (usize::MAX, 7)] {
    disjunctions.add_disjunction(group, fact);
  }

  let groups: Vec<_> = disjunctions.iter().map(|group| &group.facts).collect();
  assert_eq!(groups, [&BTreeSet::from([3, 7]), &BTreeSet::from([5, 11])]);
  assert!(disjunctions.has_conflict(&BTreeSet::from([3, 7])));
  assert!(!disjunctions.has_conflict(&BTreeSet::from([3, 5])));
}

#[test]
fn test_diff_top_k_weighted_model_count_with_multiple_exclusions() {
  let ctx = diff_top_k_proofs::DiffTopKProofsProvenance::<(), RcFamily>::new(3, true);
  let query = ctx.tagging_fn((0.5, (), None).into());
  for (weight, group) in [(0.25, 1), (0.25, 1), (0.5, 2), (0.5, 2), (0.5, 2)] {
    ctx.tagging_fn((weight, (), Some(group)).into());
  }

  // The complement is query false and exactly one fact true in each group.
  let OutputDiffProb(probability, gradient) = ctx.recover_fn(&query);
  assert_eq!(probability, 119.0 / 128.0);
  assert_eq!(
    gradient,
    [
      (0, 9.0 / 64.0),
      (1, -3.0 / 32.0),
      (2, -3.0 / 32.0),
      (3, 3.0 / 64.0),
      (4, 3.0 / 64.0),
      (5, 3.0 / 64.0)
    ]
  );
}

#[test]
fn test_sample_k_weighted_model_count_with_overlapping_proofs() {
  let ctx = diff_sample_k_proofs::DiffSampleKProofsProvenance::<(), RcFamily>::new(2);
  let a = ctx.tagging_fn((0.5, (), None).into());
  let b = ctx.tagging_fn((0.25, (), None).into());
  let c = ctx.tagging_fn((0.5, (), None).into());
  let tag = ctx.add(&ctx.mult(&a, &b), &ctx.mult(&a, &c));
  let OutputDiffProb(probability, gradient) = ctx.recover_fn(&tag);
  assert_eq!(probability, 0.3125);
  assert_eq!(gradient, [(0, 0.625), (1, 0.25), (2, 0.375)]);
}

#[test]
fn test_diff_min_max_prob_order_handles_nan_and_signed_zero() {
  use diff_min_max_prob::{Derivative, Prob};

  let tag = |p| Prob::new(p, Derivative::Zero);
  let mut tags = std::collections::BTreeSet::from([tag(f64::NAN), tag(-f64::NAN), tag(0.0), tag(-0.0)]);
  assert_eq!(tags.len(), 2);
  assert_eq!(tags.pop_first(), Some(tag(-0.0)));
  assert_eq!(tags.pop_first(), Some(tag(f64::NAN)));
}

#[test]
fn test_simple_probability_count() {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  let rt = RuntimeEnvironment::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_tagged(
      &ctx,
      vec![
        (Some(0.5), (0usize, "blue")),
        (Some(0.8), (1, "green")),
        (Some(0.2), (2, "blue")),
        (Some(0.1), (3, "green")),
        (Some(0.1), (4, "blue")),
        (Some(0.2), (5, "red")),
      ],
    );
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation_with_default_storage("_color_rev");
    strata_1.run(&ctx, &rt)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", result_1["_color_rev"].as_ref());
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(
        "count".to_string(),
        AggregateInfo::default()
          .with_arg_var_types(vec![ValueType::Str])
          .with_input_var_types(vec![ValueType::USize]),
        "_color_rev",
        ReduceGroupByType::Implicit,
      ),
    );
    strata_2.add_output_relation_with_default_storage("color_count");
    strata_2.run(&ctx, &rt)
  };

  println!("{:?}", result_2)
}

#[test]
fn test_min_max_prob_count_max() {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  let rt = RuntimeEnvironment::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_tagged(
      &ctx,
      vec![
        (Some(0.6), (0usize, "blue")),
        (Some(0.4), (0, "green")),
        (Some(0.3), (1, "blue")),
        (Some(0.7), (1, "green")),
        (Some(0.2), (2, "blue")),
        (Some(0.8), (2, "green")),
      ],
    );
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation_with_default_storage("_color_rev");
    strata_1.run(&ctx, &rt)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", result_1["_color_rev"].as_ref());
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(
        "count".to_string(),
        AggregateInfo::default().with_input_var_types(vec![ValueType::USize]),
        "_color_rev",
        ReduceGroupByType::Implicit,
      ),
    );
    strata_2.add_output_relation_with_default_storage("color_count");
    strata_2.run(&ctx, &rt)
  };

  println!("{:?}", result_2);

  let result_3 = {
    let mut strata_3 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_3.create_dynamic_relation("max_color");
    strata_3.add_input_dynamic_collection("color_count", result_2["color_count"].as_ref());
    strata_3.add_update_dataflow(
      "max_color",
      Dataflow::reduce(
        "max".to_string(),
        AggregateInfo::default()
          .with_arg_var_types(vec![ValueType::Str])
          .with_input_var_types(vec![ValueType::USize]),
        "color_count",
        ReduceGroupByType::None,
      ),
    );
    strata_3.add_output_relation_with_default_storage("max_color");
    strata_3.run(&ctx, &rt)
  };

  println!("{:?}", result_2["color_count"]);
  println!("{:?}", result_3["max_color"]);
}
