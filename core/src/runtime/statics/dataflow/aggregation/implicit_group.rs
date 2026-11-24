use std::marker::PhantomData;

use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use super::super::*;

pub struct AggregationImplicitGroup<'a, A, D, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  agg: A,
  d: D,
  rt: &'a RuntimeEnvironment,
  ctx: &'a Prov,
  phantom: PhantomData<(K, T1)>,
}

impl<'a, A, D, K, T1, Prov> AggregationImplicitGroup<'a, A, D, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  pub fn new(agg: A, d: D, rt: &'a RuntimeEnvironment, ctx: &'a Prov) -> Self {
    Self {
      agg,
      d,
      rt,
      ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, A, D, K, T1, Prov> Dataflow<(K, A::Output), Prov> for AggregationImplicitGroup<'a, A, D, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<(K, A::Output), Prov>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<(K, A::Output), Prov>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    let Some(mut batch) = self.d.iter_recent().next() else {
      return Self::Recent::empty();
    };
    // Cache the context
    let agg = self.agg;
    let rt = self.rt;
    let ctx = self.ctx;

    // Temporary function to aggregate the group and populate the result
    let consolidate_group =
      |result: &mut StaticElements<(K, A::Output), Prov>, agg_key: K, agg_group: StaticElements<T1, Prov>| {
        let agg_results = agg.aggregate(agg_group, rt, ctx);
        let joined_results = agg_results
          .into_iter()
          .map(|agg_result| StaticElement::new((agg_key.clone(), agg_result.tuple.get().clone()), agg_result.tag));
        result.extend(joined_results);
      };

    let Some(first_elem) = batch.next() else {
      return Self::Recent::empty();
    };

    // Internal states
    let mut result = vec![];
    let mut agg_key = first_elem.tuple.0.clone();
    let mut agg_group = vec![StaticElement::new(first_elem.tuple.1.clone(), first_elem.tag.clone())];

    // Enter main loop
    for curr_elem in batch {
      let curr_key = curr_elem.tuple.0.clone();
      let elem = StaticElement::new(curr_elem.tuple.1.clone(), curr_elem.tag.clone());
      if curr_key.cmp_tuple(&agg_key).is_eq() {
        agg_group.push(elem);
      } else {
        let prev_key = std::mem::replace(&mut agg_key, curr_key);
        let prev_group = std::mem::replace(&mut agg_group, vec![elem]);
        consolidate_group(&mut result, prev_key, prev_group);
      }
    }

    // Make sure we handle the last group
    consolidate_group(&mut result, agg_key, agg_group);

    // Return the result as a single batch
    Self::Recent::singleton(result.into_iter())
  }
}

impl<'a, A, D, K, T1, Prov> Clone for AggregationImplicitGroup<'a, A, D, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      rt: self.rt,
      ctx: self.ctx,
      phantom: PhantomData,
    }
  }
}
