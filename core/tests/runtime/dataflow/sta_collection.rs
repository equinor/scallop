use scallop_core::runtime::dynamic::{DynamicElement, DynamicSortedCollection};
use scallop_core::runtime::provenance::diff_min_max_prob::{Derivative, DiffMinMaxProbProvenance, Prob};
use scallop_core::runtime::provenance::*;
use scallop_core::runtime::statics::*;
use scallop_core::testing::*;
use scallop_core::utils::RcFamily;

#[test]
fn test_static_collection_dataflow_1() {
  let ctx = unit::UnitProvenance;
  let col = StaticCollection::<(i8, i8), unit::UnitProvenance>::from_vec(
    vec![
      StaticElement::new((0, 1), unit::Unit),
      StaticElement::new((1, 2), unit::Unit),
      StaticElement::new((2, 3), unit::Unit),
    ],
    &ctx,
  );
  let mut other = StaticRelation::<(i8, i8), unit::UnitProvenance>::new();
  let mut first_time = true;
  while other.changed(&ctx) || first_time {
    other.insert_dataflow_recent(&ctx, dataflow::collection(&col, first_time), true);
    first_time = false;
  }
  let result = other.complete(&ctx);
  expect_static_collection(&result, vec![(0, 1), (1, 2), (2, 3)]);
}

#[test]
fn owned_collection_iteration_preserves_tuple_order() {
  let ctx = unit::UnitProvenance;
  let collection = StaticCollection::<(i8, i8), _>::from_vec(
    vec![
      StaticElement::new((2, 3), unit::Unit),
      StaticElement::new((0, 1), unit::Unit),
    ],
    &ctx,
  );
  let mut tuples = Vec::new();
  for element in collection {
    tuples.push(*element.tuple.get());
  }
  assert_eq!(tuples, vec![(0, 1), (2, 3)]);
}

#[test]
fn merge_combines_tags_when_first_tuples_match() {
  let ctx = natural::NaturalProvenance;
  let left = StaticCollection::from_vec(vec![StaticElement::new(1_i32, 2), StaticElement::new(2, 7)], &ctx);
  let right = StaticCollection::from_vec(vec![StaticElement::new(1_i32, 3), StaticElement::new(3, 9)], &ctx);
  let result: Vec<(i32, usize)> = left.merge(right, &ctx).into_iter().map(Into::into).collect();
  assert_eq!(result, [(1, 5), (2, 7), (3, 9)]);
}

#[test]
fn overlapping_merges_keep_the_tag_order_of_the_earlier_collection() {
  let ctx = DiffMinMaxProbProvenance::<(), RcFamily>::default();
  let left = [(2_i32, 0), (3, 1)];
  let right = [(1_i32, 2), (3, 3), (4, 4)];
  let expected = [(1, 2), (2, 0), (3, 1), (4, 4)];
  let tag = |id| Prob(0.5, Derivative::Pos(id));
  let id = |tag: Prob| match tag.1 {
    Derivative::Pos(id) => id,
    _ => panic!("expected a positive derivative"),
  };

  let typed = |facts: &[(i32, usize)]| {
    StaticCollection::from_vec(
      facts.iter().map(|&(x, i)| StaticElement::new(x, tag(i))).collect(),
      &ctx,
    )
  };
  let typed_result: Vec<_> = typed(&left)
    .merge(typed(&right), &ctx)
    .into_iter()
    .map(|e| (*e.tuple.get(), id(e.tag)))
    .collect();
  assert_eq!(typed_result, expected);

  let dynamic = |facts: &[(i32, usize)]| {
    DynamicSortedCollection::from_vec(
      facts.iter().map(|&(x, i)| DynamicElement::new((x,), tag(i))).collect(),
      &ctx,
    )
  };
  let dynamic_result: Vec<_> = dynamic(&left)
    .merge(dynamic(&right), &ctx)
    .into_iter()
    .map(|e| (e.tuple[0].as_i32(), id(e.tag)))
    .collect();
  assert_eq!(dynamic_result, expected);
}
