use std::cell::Cell;

use scallop_core::common::expr::*;
use scallop_core::common::foreign_aggregate::*;
use scallop_core::common::tuple::Tuple;
use scallop_core::common::value_type::*;
use scallop_core::compiler::ram::*;
use scallop_core::runtime::dynamic::dataflow::batching::{Batch, DynamicBatch, DynamicBatches};
use scallop_core::runtime::dynamic::dataflow::{Dataflow as _, DynamicAggregationJoinGroupDataflow, DynamicDataflow};
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

fn test_group_by_key_1<Prov>() -> DynamicCollection<Prov>
where
  Prov: Provenance<InputTag = ()> + Default,
{
  let ctx = Prov::default();
  let rt = RuntimeEnvironment::new_std();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<Prov>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("colors");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.create_dynamic_relation("_colors_key");
    strata_1
      .get_dynamic_relation_unsafe("color")
      .insert_untagged(&ctx, vec![(0usize, "blue"), (1, "green"), (2, "blue")]);
    strata_1
      .get_dynamic_relation_unsafe("colors")
      .insert_untagged(&ctx, vec![("blue",), ("green",), ("red",)]);
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::relation("color".to_string()).project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_update_dataflow(
      "_colors_key",
      Dataflow::relation("colors".to_string()).project((Expr::access(0), ())),
    );
    strata_1.add_output_relation_with_default_storage("_color_rev");
    strata_1.add_output_relation_with_default_storage("_colors_key");
    strata_1.run(&ctx, &rt)
  };

  let mut result_2 = {
    let mut strata_2 = DynamicIteration::<Prov>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", result_1["_color_rev"].as_ref());
    strata_2.add_input_dynamic_collection("_colors_key", result_1["_colors_key"].as_ref());
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(
        "count".to_string(),
        AggregateInfo::default().with_input_var_types(vec![ValueType::USize]),
        "_color_rev",
        ReduceGroupByType::join("_colors_key"),
      )
      .project((Expr::access(0), Expr::access(2))),
    );
    strata_2.add_output_relation_with_default_storage("color_count");
    strata_2.run(&ctx, &rt)
  };

  result_2.remove("color_count").unwrap()
}

#[test]
fn test_group_by_key_1_unit() {
  let color_count = test_group_by_key_1::<unit::UnitProvenance>();
  expect_collection(&color_count, vec![("blue", 2usize), ("green", 1usize), ("red", 0usize)])
}

#[test]
fn test_join_group_aggregate_preserves_matching_and_empty_groups() {
  let ctx = natural::NaturalProvenance;
  let rt = RuntimeEnvironment::new_std();
  let agg = rt
    .aggregate_registry
    .instantiate_aggregator("count", AggregateInfo::default())
    .unwrap();
  let keys = vec![
    DynamicElement::new((1i32, "a"), 2),
    DynamicElement::new((1i32, "b"), 3),
    DynamicElement::new((2i32, "c"), 5),
    DynamicElement::new((2i32, "d"), 7),
    DynamicElement::new((4i32, "e"), 11),
  ];
  let values = vec![
    DynamicElement::new((0i32, 10i32), 13),
    DynamicElement::new((1i32, 10i32), 17),
    DynamicElement::new((1i32, 11i32), 19),
    DynamicElement::new((3i32, 10i32), 23),
    DynamicElement::new((4i32, 10i32), 29),
  ];
  let group = DynamicAggregationJoinGroupDataflow::new(
    agg,
    DynamicDataflow::vec(&keys),
    DynamicDataflow::vec(&values),
    &ctx,
    &rt,
  );
  let expected = [
    ((1i32, "a", 2usize), 2 * 17 * 19),
    ((1, "b", 2), 3 * 17 * 19),
    ((2, "c", 0), 5),
    ((2, "d", 0), 7),
    ((4, "e", 1), 11 * 29),
  ];
  let mut batches = group.iter_recent();
  assert!(batches
    .next()
    .unwrap()
    .map(|row| (row.tuple, row.tag))
    .eq(expected.map(|(tuple, tag)| (Tuple::from(tuple), tag))));
  assert!(batches.next().is_none());
}

#[test]
fn test_join_group_seeks_across_large_gaps_without_pulling_skipped_rows() {
  type Prov = natural::NaturalProvenance;

  #[derive(Clone)]
  struct CountedSeeks<'a>(DynamicBatch<'a, Prov>, &'a Cell<usize>);

  impl<'a> Batch<'a, Prov> for CountedSeeks<'a> {
    fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
      self.1.set(self.1.get() + 1);
      self.0.next_elem()
    }

    fn is_materialized_and_seekable(&self) -> bool {
      self.0.is_materialized_and_seekable()
    }

    fn remaining_slice(&self) -> Option<&[DynamicElement<Prov>]> {
      self.0.remaining_slice()
    }

    fn search_until(&mut self, bound: &Tuple) -> Option<DynamicElement<Prov>> {
      self.0.search_until(bound)
    }

    fn search_elem_0_until(&mut self, bound: &Tuple) -> Option<DynamicElement<Prov>> {
      self.0.search_elem_0_until(bound)
    }
  }

  impl<'a> scallop_core::runtime::dynamic::dataflow::Dataflow<'a, Prov> for CountedSeeks<'a> {
    fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
      DynamicBatches::single(self.clone())
    }
  }

  let ctx = natural::NaturalProvenance;
  let rt = RuntimeEnvironment::default();
  let agg = rt
    .aggregate_registry
    .instantiate_aggregator("count", AggregateInfo::default())
    .unwrap();
  let input =
    DynamicSortedCollection::from_vec_unchecked((0..1024i32).map(|key| DynamicElement::new((key, 0i32), 3)).collect());
  let keys = vec![
    DynamicElement::new((511i32, "a"), 2),
    DynamicElement::new((1023i32, "b"), 5),
  ];
  let pulls = Cell::new(0);
  let batch = DynamicDataflow::dynamic_recent_sorted_collection(&input)
    .iter_recent()
    .next()
    .unwrap();
  let group = DynamicAggregationJoinGroupDataflow::new(
    agg,
    DynamicDataflow::vec(&keys),
    DynamicDataflow::new(CountedSeeks(batch, &pulls)),
    &ctx,
    &rt,
  );
  assert!(group.iter_recent().flatten().map(|row| (row.tuple, row.tag)).eq([
    (Tuple::from((511i32, "a", 1usize)), 6),
    (Tuple::from((1023i32, "b", 1usize)), 15),
  ]));
  assert_eq!(pulls.get(), 2);
}
