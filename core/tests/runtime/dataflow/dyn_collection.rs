use scallop_core::runtime::dynamic::{DynamicElement, DynamicSortedCollection};
use scallop_core::runtime::provenance::diff_min_max_prob::{Derivative, DiffMinMaxProbProvenance, Prob};
use scallop_core::runtime::provenance::natural::NaturalProvenance;
use scallop_core::utils::RcFamily;

#[test]
fn merge_combines_tags_when_first_tuples_match() {
  let ctx = NaturalProvenance;
  let left = DynamicSortedCollection::from_vec(
    vec![DynamicElement::new((1_i32,), 2), DynamicElement::new((2,), 7)],
    &ctx,
  );
  let right = DynamicSortedCollection::from_vec(
    vec![DynamicElement::new((1_i32,), 3), DynamicElement::new((3,), 9)],
    &ctx,
  );
  let result: Vec<_> = left
    .merge(right, &ctx)
    .iter()
    .map(|e| (e.tuple[0].as_i32(), e.tag))
    .collect();
  assert_eq!(result, [(1, 5), (2, 7), (3, 9)]);
}

#[test]
fn overlapping_merges_keep_the_tag_order_of_the_earlier_collection() {
  let ctx = DiffMinMaxProbProvenance::<(), RcFamily>::default();
  let collection = |facts: &[(i32, usize)]| {
    DynamicSortedCollection::from_vec(
      facts
        .iter()
        .map(|&(x, id)| DynamicElement::new((x,), Prob(0.5, Derivative::Pos(id))))
        .collect(),
      &ctx,
    )
  };
  let left = collection(&[(2, 0), (3, 1)]);
  let right = collection(&[(1, 2), (3, 3), (4, 4)]);
  let result: Vec<_> = left
    .merge(right, &ctx)
    .iter()
    .map(|e| match e.tag.1 {
      Derivative::Pos(id) => (e.tuple[0].as_i32(), id),
      _ => panic!("expected a positive derivative"),
    })
    .collect();
  assert_eq!(result, [(1, 2), (2, 0), (3, 1), (4, 4)]);
}
