use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

macro_rules! compiled_fixture {
  ($program:literal) => {
    pub const SOURCE: &str = $program;
    scallop_codegen::scallop! { $program }
  };
}

fn check_parity<T, P>(
  idb: &scallop_core::runtime::database::intentional::IntentionalDatabase<P>,
  name: &str,
  actual: &scallop_core::runtime::statics::StaticOutputCollection<T, P>,
) where
  T: scallop_core::runtime::statics::StaticTupleTrait,
  P: Provenance,
  P::OutputTag: PartialEq,
{
  let mut expected = idb.get_output_collection_ref(name).unwrap().elements.clone();
  let mut actual = actual.to_dynamic_vec();
  expected.sort_by(|a, b| a.1.cmp(&b.1));
  actual.sort_by(|a, b| a.1.cmp(&b.1));
  assert_eq!(actual, expected, "{name}");
}

#[test]
fn codegen_builtin_expressions_and_grouped_aggregates_match_interpreter() {
  mod program {
    compiled_fixture! {r#"
      type inputs(i32, i32, String)
      rel inputs = {(2147483647, 0, "a"), (4, 2, "4"), (6, 2, "xyz")}
      rel computed(x + 1, y + 1, $string_length(s + s), s, s) = inputs(x, y, s)
      rel parsed(s as i32) = inputs(_, _, s)
      rel saturated(x + 1) = inputs(x, _, _)
      rel numbers(n) = n in 2..=4
      rel average(y, a) = a := avg!(x: inputs(x, y, _))
      rel largest(s, x) = (s, x) := max[s](x: inputs(x, _, s))
    "#}
  }
  let mut prov = boolean::BooleanProvenance;
  let expected = scallop_core::integrate::interpret_string_with_ctx(program::SOURCE.into(), prov.clone()).unwrap();
  let actual = program::run(&mut prov);
  check_parity(&expected, "computed", &actual.computed);
  check_parity(&expected, "parsed", &actual.parsed);
  check_parity(&expected, "saturated", &actual.saturated);
  check_parity(&expected, "numbers", &actual.numbers);
  check_parity(&expected, "average", &actual.average);
  check_parity(&expected, "largest", &actual.largest);
}

#[test]
fn codegen_entities_symbols_and_time_match_interpreter() {
  mod program {
    compiled_fixture! {r#"
      type Expression = Number(i32) | Inc(Expression)
      rel seed = {3, 5}
      rel node(new Inc(new Number(n))) = seed(n)
      rel match(n) = node(e), case e is Inc(Number(n))
      rel stamp = {t"2020-01-31T12:30:00.125Z"}
      rel offset = {d"1mo -1d 0.125s"}
      rel shifted(a + b, b) = stamp(a), offset(b)
      rel color = {s"red", s"blue"}
      rel label(s, s as String) = color(s)
    "#}
  }
  let mut prov = boolean::BooleanProvenance;
  let expected = scallop_core::integrate::interpret_string_with_ctx(program::SOURCE.into(), prov.clone()).unwrap();
  let actual = program::run(&mut prov);
  check_parity(&expected, "match", &actual.r#match);
  check_parity(&expected, "node", &actual.node);
  check_parity(&expected, "shifted", &actual.shifted);
  check_parity(&expected, "label", &actual.label);
}

#[test]
fn codegen_tagged_rules_and_exclusion_match_interpreter() {
  mod program {
    compiled_fixture! {r#"
      rel variable = {1, 2}
      rel { assignment(x, true); assignment(x, false) } = variable(x)
      type denominator(i32, f32)
      rel denominator = {(1, 1.0), (2, 2.0)}
      rel (1.0 / p)::weighted(x, b) = assignment(x, b), denominator(x, p)
      rel impossible(x) = assignment(x, true), assignment(x, false)
    "#}
  }
  let mut prov = top_k_proofs::TopKProofsProvenance::<scallop_core::utils::RcFamily>::new(8, false);
  let expected = scallop_core::integrate::interpret_string_with_ctx(program::SOURCE.into(), prov.clone()).unwrap();
  let actual = program::run(&mut prov);
  check_parity(&expected, "weighted", &actual.weighted);
  check_parity(&expected, "impossible", &actual.impossible);
}

#[test]
fn codegen_indexed_and_wide_relations_match_interpreter() {
  mod program {
    compiled_fixture! {r#"
      @storage("indexed_vec")
      type indexed(i: usize, n: i32)
      rel indexed = {(0, 7), (0, 8), (2, 9)}
      type key(usize)
      rel key = {0, 1, 2}
      rel lookup(i, n) = key(i), indexed(i, n)
      rel six = {(1, 2, 3, 4, 5, 6)}
      rel wide(a, b, c, d, e, f, a, b, c, d, e, f, 13) = six(a, b, c, d, e, f)
      rel projected(x) = wide(_, _, _, _, _, _, _, _, _, _, _, _, x)
    "#}
  }
  let mut prov = boolean::BooleanProvenance;
  let expected = scallop_core::integrate::interpret_string_with_ctx(program::SOURCE.into(), prov.clone()).unwrap();
  let actual = program::run(&mut prov);
  check_parity(&expected, "indexed", &actual.indexed);
  check_parity(&expected, "lookup", &actual.lookup);
  check_parity(&expected, "wide", &actual.wide);
  check_parity(&expected, "projected", &actual.projected);
}

#[test]
fn codegen_scheduler_and_goal_match_interpreter() {
  use scallop_core::integrate::{IntegrateOptions, InterpretContext};
  use scallop_core::runtime::env::{RuntimeEnvironmentOptions, Scheduler};

  mod program {
    compiled_fixture! {r#"
      rel edge = {0.2::(0, 1), 0.9::(0, 2), 0.8::(2, 1), 0.7::(2, 3), 0.5::(3, 4)}
      @scheduler("astar")
      rel path(a, b) = edge(a, b) or (path(a, c), edge(c, b))
      @goal
      rel goal() = path(0, 1)
    "#}
  }
  let mut prov = min_max_prob::MinMaxProbProvenance::new();
  let options = RuntimeEnvironmentOptions::default()
    .with_stop_when_goal_non_empty(true)
    .with_default_scheduler(Some(Scheduler::Beam { beam_size: 2 }));
  let mut interpreter = InterpretContext::<_, scallop_core::utils::RcFamily>::new_with_options(
    program::SOURCE.into(),
    prov.clone(),
    IntegrateOptions {
      runtime_environment_options: options.clone(),
      ..Default::default()
    },
  )
  .unwrap();
  interpreter.run().unwrap();
  let actual = program::try_run_with_edb(&mut prov, program::create_edb(), options).unwrap();
  let expected = interpreter.idb();
  check_parity(&expected, "path", &actual.path);
  check_parity(&expected, "goal", &actual.goal);
}

#[test]
fn codegen_edge_path_left_recursion() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = path(a, b) and edge(b, c)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = edge_path::run(&mut ctx);
  expect_static_output_collection(&result.edge, vec![(0, 1), (1, 2)]);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn codegen_edge_path_right_recursion() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = edge(a, b) and path(b, c)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = edge_path::run(&mut ctx);
  expect_static_output_collection(&result.edge, vec![(0, 1), (1, 2)]);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn codegen_edge_path_binary_recursion() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = path(a, b) and path(b, c)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = edge_path::run(&mut ctx);
  expect_static_output_collection(&result.edge, vec![(0, 1), (1, 2)]);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn codegen_odd_even() {
  mod odd_even {
    use scallop_codegen::scallop;
    scallop! {
      rel even(0)
      rel odd(x) = even(x - 1), x < 10
      rel even(x) = odd(x - 1), x < 10
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = odd_even::run(&mut ctx);
  expect_static_output_collection(&result.odd, vec![(1,), (3,), (5,), (7,), (9,)]);
  expect_static_output_collection(&result.even, vec![(0,), (2,), (4,), (6,), (8,)]);
}

#[test]
fn codegen_difference_1() {
  mod difference_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel a = {(0, 1), (1, 2)}
      rel b = {(1, 1), (1, 2)}
      rel s(x, y) = a(x, y), ~b(x, y)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = difference_1::run(&mut ctx);
  expect_static_output_collection(&result.s, vec![(0, 1)]);
}

#[test]
fn codegen_bmi_test_1() {
  mod bmi_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel student = {
        (1, 185, 80, "Mary"),
        (2, 175, 70, "John"),
        (3, 165, 55, "Maomao"),
      }

      rel height(id, h) = student(id, h, _, _)
      rel weight(id, w) = student(id, _, w, _)

      rel bmi(id, w as f32 / ((h * h) as f32 / 10000.0)) = height(id, h), weight(id, w)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = bmi_test_1::run(&mut ctx);
  expect_static_output_collection(&result.height, vec![(1, 185), (2, 175), (3, 165)]);
  expect_static_output_collection(&result.weight, vec![(1, 80), (2, 70), (3, 55)]);
  expect_static_output_collection(&result.bmi, vec![(1, 23.374), (2, 22.857), (3, 20.202)]);
}

#[test]
fn codegen_bmi_test_2() {
  mod bmi_test_2 {
    use scallop_codegen::scallop;
    scallop! {
      type student(usize, f32, f32, String)

      rel student = {
        (1, 185, 80, "Mary"),
        (2, 175, 70, "John"),
        (3, 165, 55, "Maomao"),
      }

      rel height(id, h) = student(id, h, _, _)
      rel weight(id, w) = student(id, _, w, _)

      rel bmi(id, w / (h * h / 10000.0)) = height(id, h), weight(id, w)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = bmi_test_2::run(&mut ctx);
  expect_static_output_collection(&result.height, vec![(1, 185.0), (2, 175.0), (3, 165.0)]);
  expect_static_output_collection(&result.weight, vec![(1, 80.0), (2, 70.0), (3, 55.0)]);
  expect_static_output_collection(&result.bmi, vec![(1, 23.374), (2, 22.857), (3, 20.202)]);
}

#[test]
fn codegen_const_fold_test_1() {
  mod const_fold_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel E(1)
      rel R(s, a) = s == x + z, x == y + 1, y == z + 1, z == 1, E(a)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = const_fold_test_1::run(&mut ctx);
  expect_static_output_collection(&result.R, vec![(4i32, 1i32)]);
}

#[test]
fn codegen_digit_sum_test_1() {
  mod digit_sum_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel digit = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
      }
      rel sum_2(0, 1, x + y) = digit(0, x), digit(1, y)
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = digit_sum_test_1::run(&mut ctx);
  expect_static_output_collection(
    &result.sum_2,
    vec![
      (0, 1, 0),
      (0, 1, 1),
      (0, 1, 2),
      (0, 1, 3),
      (0, 1, 4),
      (0, 1, 5),
      (0, 1, 6),
      (0, 1, 7),
      (0, 1, 8),
    ],
  );
}

#[test]
fn codegen_digit_sum_test_2() {
  mod digit_sum_test_2 {
    use scallop_codegen::scallop;
    scallop! {
      rel digit = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
      }
      rel sum_2(a, b, c) = digit(a, x), digit(b, y), c == x + y, a == 0, b == 1
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = digit_sum_test_2::run(&mut ctx);
  expect_static_output_collection(
    &result.sum_2,
    vec![
      (0, 1, 0),
      (0, 1, 1),
      (0, 1, 2),
      (0, 1, 3),
      (0, 1, 4),
      (0, 1, 5),
      (0, 1, 6),
      (0, 1, 7),
      (0, 1, 8),
    ],
  );
}

#[test]
fn codegen_expr_test_1() {
  mod expr_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel eval(e, c) = constant(e, c)
      rel eval(e, a + b) = binary(e, "+", l, r), eval(l, a), eval(r, b)
      rel eval(e, a - b) = binary(e, "-", l, r), eval(l, a), eval(r, b)
      rel result(y) = eval(e, y), goal(e)

      rel constant = { (0, 1), (1, 2), (2, 3) }
      rel binary = { (3, "+", 0, 1), (4, "-", 3, 2) }
      rel goal(4)
      query result
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = expr_test_1::run(&mut ctx);
  expect_static_output_collection(&result.result, vec![(0,)])
}

#[test]
fn codegen_fib_test_1() {
  mod fib_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel fib :- {(0, 1), (1, 1)}
      rel fib(x, a + b) :- fib(x - 1, a), fib(x - 2, b), x <= 7
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = fib_test_1::run(&mut ctx);
  expect_static_output_collection(
    &result.fib,
    vec![(0i32, 1i32), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (6, 13), (7, 21)],
  );
}

#[test]
fn codegen_count_edge_1() {
  mod count_edge {
    use scallop_codegen::scallop;
    scallop! {
      type edge(usize, usize)
      rel edge :- {(0, 1), (1, 2), (2, 3)}
      rel num_edges(n) :- n = count(a, b: edge(a, b))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = count_edge::run(&mut ctx);
  expect_static_output_collection(&result.num_edges, vec![(3,)]);
}

#[test]
fn codegen_out_degree_1() {
  mod out_degree {
    use scallop_codegen::scallop;
    scallop! {
      rel edge :- {(0, 1), (1, 2)}
      rel out_degree(x, n) :- n = count(y: edge(x, y))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = out_degree::run(&mut ctx);
  expect_static_output_collection(&result.out_degree, vec![(0, 1), (1, 1)]);
}

#[test]
fn codegen_out_degree_2() {
  mod out_degree {
    use scallop_codegen::scallop;
    scallop! {
      rel node :- {0, 1, 2}
      rel edge :- {(0, 1), (1, 2)}
      rel out_degree(x, n) :- n = count(y: edge(x, y) where x: node(x))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = out_degree::run(&mut ctx);
  expect_static_output_collection(&result.out_degree, vec![(0, 1), (1, 1), (2, 0)]);
}

#[test]
fn codegen_exists_1() {
  mod exists_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(x, y) = edge(x, y) or (path(x, z) and edge(z, y))
      rel result1(b) = b := exists(path(0, 2))
      rel result2(b) = b := exists(path(0, 3))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = exists_1::run(&mut ctx);
  expect_static_output_collection(&result.result1, vec![(true,)]);
  expect_static_output_collection(&result.result2, vec![(false,)]);
}

#[test]
fn codegen_sum_1() {
  mod sum_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel color_num_obj :- {("blue", 1), ("red", 3), ("yellow", 6)}
      rel num_obj(n) :- n = sum(y: color_num_obj(_, y))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = sum_1::run(&mut ctx);
  expect_static_output_collection(&result.num_obj, vec![(10,)]);
}

#[test]
fn codegen_max_1() {
  mod max_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel color_num_obj :- {("blue", 1), ("red", 3), ("yellow", 6)}
      rel max_num_per_color(n) :- n = max(y: color_num_obj(_, y))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = max_1::run(&mut ctx);
  expect_static_output_collection(&result.max_num_per_color, vec![(6,)]);
}

#[test]
fn codegen_min_1() {
  mod min_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel color_num_obj :- {("blue", 1), ("red", 3), ("yellow", 6)}
      rel min_num_per_color(n) :- n = min(y: color_num_obj(_, y))
    }
  }

  let mut ctx = unit::UnitProvenance;
  let result = min_1::run(&mut ctx);
  expect_static_output_collection(&result.min_num_per_color, vec![(1,)]);
}

#[test]
fn codegen_simple_test_1() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2)]);
}

#[test]
fn codegen_simple_test_2() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2), (2, 2)}
      rel self_edge(a, a) :- edge(a, a)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.self_edge, vec![(2, 2)]);
}

#[test]
fn codegen_simple_test_3() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel something(a, 2) :- edge(a, b)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.something, vec![(0, 2), (1, 2)]);
}

#[test]
fn codegen_simple_test_4() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel something(a, 2) :- edge(a, b), b > 1
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.something, vec![(1, 2)]);
}

#[test]
fn codegen_simple_test_5() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(1, 2), (2, 3), (3, 4)}
      rel R = {(1, 2), (4, 3)}
      rel O(a, b) = S(b, a), R(a, b)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(4, 3)]);
}

#[test]
fn codegen_simple_test_6() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(1, 2), (2, 3), (3, 4), (4, 3)}
      rel R = {(1, 2), (3, 4), (4, 3)}
      rel O(a, b) = S(b, a), S(a, b), R(a, b), R(b, a)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(3, 4), (4, 3)]);
}

#[test]
fn codegen_simple_test_7() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1), (2)}
      rel O(a, b) = S(a, b), ~R(b)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(2, 3)]);
}

#[test]
fn codegen_simple_test_8() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1, 2), (2, 3)}
      rel O(a, b) = S(a, b), ~R(b, c)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(2, 3)]);
}

#[test]
fn codegen_simple_test_9() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1, 2), (2, 3), (2, 2)}
      rel O(a, b) = S(a, b), ~R(a, a)
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(0, 1), (1, 2)]);
}

#[test]
fn codegen_srl_1() {
  mod srl_1_test {
    use scallop_codegen::scallop;
    scallop! {
      rel verb = {
        (1, "play"),
        (4, "play"),
      }

      rel noun = {
        (2, "Tom", "Person"),
        (3, "Jerry", "Person")
      }

      rel arg1 = {
        (1, 2),
        (1, 3),
      }

      rel synonym = {
        (1, 4),
      }

      rel plays_soccer(n) =
        verb(vid0, "play"),
        arg1(vid, n),
        noun(n, _, "Person"),
        synonym(vid, vid0)

      rel how_many_play_soccer(c) = c := count(n: plays_soccer(n))
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = srl_1_test::run(&mut ctx);
  expect_static_output_collection(&result.how_many_play_soccer, vec![(2,)]);
}

#[test]
fn codegen_class_student_grade_1() {
  mod srl_1_test {
    use scallop_codegen::scallop;
    scallop! {
      rel class_student_grade = {
        (0, "tom", 50),
        (0, "jerry", 70),
        (0, "alice", 60),
        (1, "bob", 80),
        (1, "sherry", 90),
        (1, "frank", 30),
      }

      rel class_top_student(c, s) = s := argmax[s](g: class_student_grade(c, s, g))
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = srl_1_test::run(&mut ctx);
  expect_static_output_collection(
    &result.class_top_student,
    vec![(0, "jerry".to_string()), (1, "sherry".to_string())],
  );
}

#[test]
fn codegen_remove_unused_relation() {
  mod test {
    use scallop_codegen::scallop;
    scallop! {
      rel digit_1(1)
      rel digit_2(3)
      rel sum_2(a + b) = digit_1(a) and digit_2(b)
      rel mult_2(a * b) = digit_1(a) and digit_2(b)
      query sum_2
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = test::run(&mut ctx);
  expect_static_output_collection(&result.sum_2, vec![(4,)]);
}

#[test]
fn codegen_negate_query_1() {
  mod test {
    use scallop_codegen::scallop;
    scallop! {
      rel A = {(0, 1)}
      rel B = {(0, 1, "O")}
      rel Q() = A(a, b), ~B(a, b, "O")
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = test::run(&mut ctx);
  expect_static_output_collection(&result.Q, vec![]);
}

#[test]
fn codegen_negate_query_2() {
  mod test {
    use scallop_codegen::scallop;
    scallop! {
      rel B("Alice")
      rel A() :- ~B(_)
      query A
    }
  }
  let mut ctx = unit::UnitProvenance;
  let result = test::run(&mut ctx);
  expect_static_output_collection(&result.A, vec![]);
}
