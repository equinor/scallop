use std::cell::Cell;
use std::collections::BTreeSet;
use std::rc::Rc;

use scallop_core::common::expr::*;
use scallop_core::common::foreign_aggregate::*;
use scallop_core::common::value_type::ValueType;
use scallop_core::compiler::ram::*;
use scallop_core::integrate::IntegrateContext;
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
fn test_weighted_model_count_for_overlapping_proofs() {
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

#[derive(Clone)]
struct OneVariableDdBackend(Rc<Cell<usize>>);

struct OneVariableDd {
  fact: usize,
  low: bool,
  high: bool,
}

impl WmcBackend for OneVariableDdBackend {
  fn wmc<S: Semiring>(
    &self,
    formula: &BooleanFormula,
    semiring: &S,
    weight: &impl Fn(&usize) -> S::Element,
  ) -> S::Element {
    self.0.set(self.0.get() + 1);
    fn evaluate(formula: &BooleanFormula, value: bool) -> bool {
      match formula {
        BooleanFormula::Constant(b) => *b,
        BooleanFormula::Literal(lit) => lit.sign() == value,
        BooleanFormula::Not(term) => !evaluate(term, value),
        BooleanFormula::And(terms) => terms.iter().all(|term| evaluate(term, value)),
        BooleanFormula::Or(terms) => terms.iter().any(|term| evaluate(term, value)),
      }
    }

    let &[fact] = formula.fact_ids().as_slice() else {
      panic!("test adapter supports one variable");
    };
    let diagram = OneVariableDd {
      fact,
      low: evaluate(formula, false),
      high: evaluate(formula, true),
    };
    match (diagram.low, diagram.high) {
      (false, false) => semiring.zero(),
      (true, true) => semiring.one(),
      (true, false) => semiring.negate(weight(&diagram.fact)),
      (false, true) => weight(&diagram.fact),
    }
  }
}

#[test]
fn test_custom_decision_diagram_backend_for_differentiable_provenance() {
  let calls = Rc::new(Cell::new(0));
  let ctx = diff_top_k_proofs::DiffTopKProofsProvenance::<(), RcFamily, _>::new_with_wmc_backend(
    3,
    true,
    OneVariableDdBackend(calls.clone()),
  );
  let tag = ctx.tagging_fn((0.25, (), Some(4)).into());
  assert_eq!(ctx.weight(&tag), 0.25);
  let OutputDiffProb(probability, gradient) = ctx.clone().recover_fn(&tag);
  // With a singleton exclusion, the query or the exactly-one violation is a tautology.
  assert_eq!(probability, 1.0);
  assert!(gradient.is_empty());
  assert_eq!(calls.get(), 2);

  let tag = Literal::Neg(usize::MAX);
  assert_eq!(
    tag.wmc_using(&OneVariableDdBackend(calls.clone()), &RealSemiring, &|id| {
      assert_eq!(*id, usize::MAX);
      0.25
    }),
    0.75
  );
  assert_eq!(calls.get(), 3);
}

#[test]
fn test_weighted_model_count_does_not_reuse_weights_between_queries() {
  let first = DNFFormula::new(vec![Clause::new(vec![Literal::Pos(7), Literal::Neg(20)])]);
  let second = DNFFormula::new(vec![Clause::new(vec![Literal::Pos(100), Literal::Neg(usize::MAX)])]);
  assert_eq!(first.wmc(&RealSemiring, &|_| 0.5), 0.25);
  assert_eq!(second.wmc(&RealSemiring, &|_| 0.25), 0.1875);
  assert_eq!(first.wmc(&RealSemiring, &|_| 0.125), 0.109375);
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
fn test_top_k_proofs_breaks_addition_and_join_ties_by_literals() {
  let ctx = diff_top_k_proofs::DiffTopKProofsProvenance::<(), RcFamily>::new(1, false);
  let first = ctx.tagging_fn((0.5, (), None).into());
  let second = ctx.tagging_fn((0.5, (), None).into());
  let third = ctx.tagging_fn((0.5, (), None).into());
  let first_and_third = ctx.mult(&first, &third);

  for (left, right) in [(&first, &second), (&second, &first)] {
    let winner = ctx.add(left, right);
    assert_eq!(winner, first);
    let OutputDiffProb(score, gradient) = ctx.recover_fn(&winner);
    assert_eq!((score, gradient.as_slice()), (0.5, &[(0, 1.0)][..]));

    let candidates = left.or(right);
    assert_eq!(ctx.mult(&candidates, &third), first_and_third);
    assert_eq!(ctx.mult(&third, &candidates), first_and_third);
  }
}

#[test]
fn test_top_k_proofs_breaks_negation_frontier_ties_by_literals() {
  let ctx = top_k_proofs::TopKProofsProvenance::<RcFamily>::new(2, false);
  let [a, b, c, d] = std::array::from_fn(|_| ctx.tagging_fn((0.5, None).into()));
  let ab = ctx.mult(&a, &b);
  let cd = ctx.mult(&c, &d);
  for (left, right) in [(&ab, &cd), (&cd, &ab)] {
    let result = ctx.negate(&left.or(right)).unwrap();
    assert_eq!(
      result.clauses,
      [
        Clause::new(vec![Literal::Neg(0), Literal::Neg(2)]),
        Clause::new(vec![Literal::Neg(0), Literal::Neg(3)]),
      ]
    );
  }
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
fn test_diff_min_max_prob_selects_tied_derivatives_independent_of_operand_order() {
  use diff_min_max_prob::*;

  let ctx = DiffMinMaxProbProvenance::<(), RcFamily>::default();
  let tag = |p| ctx.tagging_fn((p, Some(())).into());
  let derivative = |tag: Prob| {
    let OutputDiffProb(_, id, sign, _) = ctx.recover_fn(&tag);
    (id, sign)
  };
  let first = tag(0.5);
  let second = tag(0.5);
  let negated = ctx.negate(&second).unwrap();
  for (left, right) in [(&first, &second), (&second, &first)] {
    assert_eq!(derivative(ctx.add(left, right)), (1, 1));
    assert_eq!(derivative(ctx.mult(left, right)), (0, 1));
  }
  for (left, right) in [(&second, &negated), (&negated, &second)] {
    assert_eq!(derivative(ctx.add(left, right)), (1, 1));
    assert_eq!(derivative(ctx.mult(left, right)), (1, -1));
  }

  let zero_score = tag(0.0);
  let one_score = tag(1.0);
  assert_eq!(derivative(ctx.add(&zero_score, &ctx.zero())), (2, 1));
  assert_eq!(derivative(ctx.mult(&ctx.one(), &one_score)), (3, 1));
}

#[test]
fn test_diff_min_max_prob_reordering_rules_preserves_gradients() {
  use diff_min_max_prob::*;

  for (first, second) in [("input_a", "input_b"), ("input_b", "input_a")] {
    let mut ctx = IntegrateContext::<_, RcFamily>::new(DiffMinMaxProbProvenance::<(), RcFamily>::default());
    ctx
      .add_program(&format!(
        r#"
          type {first}(i32)
          type {second}(i32)
          rel alternative(x) = {first}(x)
          rel alternative(x) = {second}(x)
          rel joined(x) = {first}(x) and {second}(x)
          query input_a
          query input_b
          query alternative
          query joined
        "#
      ))
      .unwrap();
    for relation in [first, second] {
      ctx
        .add_facts(relation, vec![(Some((0.5, Some(())).into()), (1_i32,).into())], true)
        .unwrap();
    }
    ctx.run().unwrap();

    for (relation, expected_id) in [("input_a", 0), ("input_b", 1), ("alternative", 1), ("joined", 0)] {
      let output = ctx.computed_relation_ref(relation).unwrap();
      let [(OutputDiffProb(score, id, sign, _), _)] = output.elements.as_slice() else {
        panic!("expected one result for {relation}");
      };
      assert_eq!((*score, *id, *sign), (0.5, expected_id, 1), "{relation}");
    }
  }
}

#[test]
fn test_diff_min_max_prob_propagates_late_equal_score_gradient() {
  use diff_min_max_prob::*;

  let ctx = DiffMinMaxProbProvenance::<(), RcFamily>::default();
  let mut iteration = DynamicIteration::new();
  for relation in ["early", "late", "delay", "merged", "output"] {
    iteration.create_dynamic_relation(relation);
  }
  for relation in ["early", "late"] {
    iteration
      .get_dynamic_relation_unsafe(relation)
      .insert_one_tagged(&ctx, Some((0.5, Some(())).into()), (1_i32,));
  }
  iteration.add_update_dataflow("merged", Dataflow::relation("early"));
  iteration.add_update_dataflow("delay", Dataflow::relation("late"));
  iteration.add_update_dataflow("merged", Dataflow::relation("delay"));
  iteration.add_update_dataflow("output", Dataflow::relation("merged"));
  iteration.add_output_relation_with_default_storage("output");

  let runtime = RuntimeEnvironment::default();
  let result = iteration.run(&ctx, &runtime);
  let mut output = result["output"].iter();
  let OutputDiffProb(score, id, sign, _) = ctx.recover_fn(&output.next().unwrap().tag);
  assert_eq!((score, id, sign), (0.5, 1, 1));
  assert!(output.next().is_none());
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
