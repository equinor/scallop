use scallop_core::{
  runtime::dynamic::{DynamicElement, DynamicSortedCollection},
  runtime::{env::RuntimeEnvironmentOptions, env::Scheduler, provenance},
  testing::*,
  utils::RcFamily,
};

#[test]
fn prob_edge_path_with_goal_and_a_star_search() {
  expect_interpret_within_iter_limit_with_ctx_and_runtime_options::<
    provenance::min_max_prob::MinMaxProbProvenance,
    RcFamily,
  >(
    r#"
      rel edge = {
        0.2::(0, 100),
        0.2::(100, 101),
        0.2::(101, 1000),

        0.99::(0, 200),
        0.99::(200, 201),
        0.99::(201, 202),
        0.99::(202, 203),
        0.99::(203, 1000),

        0.98::(1000, 2000),
        0.97::(2000, 3000),
        0.96::(3000, 4000),
      }

      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)

      @goal
      rel goal() = path(0, 1000)
    "#,
    9,
    provenance::min_max_prob::MinMaxProbProvenance::new(),
    RuntimeEnvironmentOptions::default()
      .with_stop_when_goal_non_empty(true)
      .with_default_scheduler(Some(Scheduler::AStar)),
  );
}

#[test]
fn beam_ranking_preserves_weight_ties_and_waitlist_chronology() {
  let ctx = provenance::add_mult_prob::AddMultProbProvenance::default();
  let scheduler = Scheduler::Beam { beam_size: 3 };
  let make_delta = |rows: &[(i32, f64)]| {
    DynamicSortedCollection::from_vec_unchecked(
      rows
        .iter()
        .map(|&(tuple, tag)| DynamicElement::new(tuple, tag))
        .collect(),
    )
  };
  let mut delta = make_delta(&[(0, 0.6), (1, 0.1), (2, 0.6), (3, -0.0), (4, 0.0), (5, 0.9), (6, 0.6)]);
  let mut waitlist = vec![DynamicElement::new(-1i32, 0.99)];
  let tuples = |rows: &[DynamicElement<_>]| rows.iter().map(|row| row.tuple.as_i32()).collect::<Vec<_>>();
  scheduler.schedule(&mut delta, &mut waitlist, &mut [], &ctx);
  assert_eq!(tuples(&delta), [0, 2, 5]);
  assert_eq!(tuples(&waitlist), [-1, 4, 3, 1, 6]);

  let mut delta = make_delta(&[(8, 0.2)]);
  scheduler.schedule(&mut delta, &mut waitlist, &mut [], &ctx);
  assert_eq!(tuples(&delta), [1, 6, 8]);
  assert_eq!(tuples(&waitlist), [-1, 4, 3]);
}

#[test]
fn beam_ranking_preserves_unordered_priority_behavior() {
  let ctx = provenance::add_mult_prob::AddMultProbProvenance::default();
  let scheduler = Scheduler::Beam { beam_size: 2 };
  let mut delta = DynamicSortedCollection::from_vec_unchecked(vec![
    DynamicElement::new(0i32, 0.4),
    DynamicElement::new(1i32, f64::NAN),
    DynamicElement::new(2i32, 0.7),
    DynamicElement::new(3i32, 0.2),
  ]);
  let mut waitlist = Vec::new();
  scheduler.schedule(&mut delta, &mut waitlist, &mut [], &ctx);
  assert!(delta.elements.iter().map(|row| row.tuple.as_i32()).eq([0, 1]));
  assert!(waitlist.iter().map(|row| row.tuple.as_i32()).eq([3, 2]));
}
