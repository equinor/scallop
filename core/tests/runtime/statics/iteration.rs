use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::runtime::statics::*;
use scallop_core::testing::*;

#[test]
fn test_static_iter_edge_path() {
  let env = RuntimeEnvironment::default();
  let mut prov = unit::UnitProvenance;
  let mut iter = StaticIteration::<unit::UnitProvenance>::new(&env, &mut prov);

  // Add relations
  let edge = iter.create_relation::<(usize, usize)>();
  let path = iter.create_relation::<(usize, usize)>();
  let path_inv = iter.create_relation::<(usize, usize)>();

  // Add facts
  edge.insert_untagged(iter.provenance_context, vec![(0, 1), (1, 2)]);

  // Main loop
  while iter.changed() || iter.is_first_iteration() {
    iter.insert_dataflow(&path, &edge);
    iter.insert_dataflow(&path_inv, dataflow::project(&path, |t| (t.1, t.0)));
    iter.insert_dataflow(&path, dataflow::project(iter.join(&path_inv, &edge), |t| (t.1, t.2)));
    iter.step();
  }

  // Peek the path
  let path = iter.complete(&path);
  expect_static_collection(&path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn test_static_join_uses_collection_order_for_float_keys() {
  let env = RuntimeEnvironment::default();
  let mut prov = unit::UnitProvenance;
  let mut iter = StaticIteration::new(&env, &mut prov);
  let left = iter.create_relation::<((f64,), i32)>();
  let right = iter.create_relation::<((f64,), i32)>();
  let joined = iter.create_relation::<(i32, i32)>();
  let found = iter.create_relation::<i32>();

  left.insert_untagged(
    iter.provenance_context,
    vec![((f64::NAN,), 1), ((-0.0,), 2), ((1.0,), 5), ((-f64::NAN,), 6)],
  );
  right.insert_untagged(iter.provenance_context, vec![((0.0,), 3), ((-f64::NAN,), 4)]);
  while iter.changed() || iter.is_first_iteration() {
    iter.insert_dataflow(&joined, dataflow::project(iter.join(&left, &right), |t| (t.1, t.2)));
    iter.insert_dataflow(&found, dataflow::project(dataflow::find(&left, (f64::NAN,)), |t| t.1));
    iter.step();
  }
  expect_static_collection(&iter.complete(&joined), vec![(1, 4), (2, 3), (6, 4)]);
  expect_static_collection(&iter.complete(&found), vec![1, 6]);
}

#[test]
fn test_static_iter_odd_even_3() {
  let env = RuntimeEnvironment::default();
  let mut prov = unit::UnitProvenance;

  struct Stratum0Result<C: Provenance> {
    numbers: StaticCollection<(i32,), C>,
    _numbers_perm_0_: StaticCollection<(i32, ()), C>,
    _numbers_perm_0_0: StaticCollection<(i32, i32), C>,
  }

  fn stratum_0<C: Provenance>(env: &RuntimeEnvironment, prov: &mut C) -> Stratum0Result<C> {
    let mut iter = StaticIteration::<C>::new(env, prov);
    let numbers = iter.create_relation::<(i32,)>();
    let _numbers_perm_0_ = iter.create_relation::<(i32, ())>();
    let _numbers_perm_0_0 = iter.create_relation::<(i32, i32)>();
    numbers.insert_untagged(iter.provenance_context, vec![(0,)]);
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &numbers,
        dataflow::project(
          dataflow::project(
            dataflow::filter(dataflow::project(&numbers, |t| (t.0 + 1,)), |t| t.0 <= 10),
            |t| t.0,
          ),
          |t| (t,),
        ),
      );
      iter.insert_dataflow(&_numbers_perm_0_0, dataflow::project(&numbers, |t| (t.0, t.0)));
      iter.insert_dataflow(&_numbers_perm_0_, dataflow::project(&numbers, |t| (t.0, ())));
      iter.step();
    }
    Stratum0Result {
      numbers: iter.complete(&numbers),
      _numbers_perm_0_: iter.complete(&_numbers_perm_0_),
      _numbers_perm_0_0: iter.complete(&_numbers_perm_0_0),
    }
  }

  struct Stratum1Result<C: Provenance> {
    odd: StaticCollection<(i32,), C>,
  }

  fn stratum_1<C: Provenance>(
    env: &RuntimeEnvironment,
    prov: &mut C,
    stratum_0_result: &Stratum0Result<C>,
  ) -> Stratum1Result<C> {
    let mut iter = StaticIteration::<C>::new(env, prov);
    let _temp_0 = iter.create_relation::<(i32, i32)>();
    let odd = iter.create_relation::<(i32,)>();
    let _odd_perm_0 = iter.create_relation::<(i32, ())>();
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &_temp_0,
        dataflow::project(
          dataflow::collection(&stratum_0_result._numbers_perm_0_0, iter.is_first_iteration()),
          |t| (t.0 - 2, t.0),
        ),
      );
      iter.insert_dataflow(
        &odd,
        dataflow::project(
          dataflow::project(
            dataflow::find(
              dataflow::collection(&stratum_0_result._numbers_perm_0_, iter.is_first_iteration()),
              1,
            ),
            |t| (t,),
          ),
          |_| (1,),
        ),
      );
      iter.insert_dataflow(
        &odd,
        dataflow::project(dataflow::project(iter.join(&_temp_0, &_odd_perm_0), |t| t.1), |t| (t,)),
      );
      iter.insert_dataflow(&_odd_perm_0, dataflow::project(&odd, |t| (t.0, ())));
      iter.step();
    }
    Stratum1Result {
      odd: iter.complete(&odd),
    }
  }

  struct Stratum2Result<C: Provenance> {
    even: StaticCollection<(i32,), C>,
  }

  fn stratum_2<C: Provenance>(
    env: &RuntimeEnvironment,
    prov: &mut C,
    stratum_0_result: &Stratum0Result<C>,
    stratum_1_result: &Stratum1Result<C>,
  ) -> Stratum2Result<C> {
    let mut iter = StaticIteration::<C>::new(env, prov);
    let even = iter.create_relation::<(i32,)>();
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &even,
        dataflow::project(
          dataflow::project(
            iter.difference(
              dataflow::collection(&stratum_0_result.numbers, iter.is_first_iteration()),
              dataflow::collection(&stratum_1_result.odd, iter.is_first_iteration()),
            ),
            |t| t.0,
          ),
          |t| (t,),
        ),
      );
      iter.step();
    }
    Stratum2Result {
      even: iter.complete(&even),
    }
  }

  // Execute
  let stratum_0_result = stratum_0(&env, &mut prov);
  let stratum_1_result = stratum_1(&env, &mut prov, &stratum_0_result);
  let stratum_2_result = stratum_2(&env, &mut prov, &stratum_0_result, &stratum_1_result);

  // Check result
  expect_static_collection(&stratum_1_result.odd, vec![(1,), (3,), (5,), (7,), (9,)]);
  expect_static_collection(&stratum_2_result.even, vec![(0,), (2,), (4,), (6,), (8,), (10,)]);
}
