use crate::common::tuple::*;

use super::*;

pub struct DynamicAggregationImplicitGroupDataflow<'a, Prov: Provenance> {
  pub agg: DynamicAggregator<Prov>,
  pub d: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Clone for DynamicAggregationImplicitGroupDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      ctx: self.ctx,
      runtime: self.runtime,
    }
  }
}

impl<'a, Prov: Provenance> DynamicAggregationImplicitGroupDataflow<'a, Prov> {
  pub fn new(
    agg: DynamicAggregator<Prov>,
    d: DynamicDataflow<'a, Prov>,
    ctx: &'a Prov,
    runtime: &'a RuntimeEnvironment,
  ) -> Self {
    Self { agg, d, ctx, runtime }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicAggregationImplicitGroupDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let Some(mut batch) = self.d.iter_recent().next_batch() else {
      return DynamicBatches::empty();
    };
    // Temporary function to aggregate the group and populate the result
    let consolidate_group = |result: &mut DynamicElements<Prov>, agg_key: Tuple, agg_group| {
      let agg_results = self.agg.aggregate(self.ctx, self.runtime, agg_group);
      let joined_results = agg_results
        .into_iter()
        .map(|agg_result| DynamicElement::new((agg_key.clone(), agg_result.tuple.clone()), agg_result.tag));
      result.extend(joined_results);
    };

    let Some(first_elem) = batch.next_elem() else {
      return DynamicBatches::empty();
    };

    // Internal states
    let mut result = vec![];
    let mut agg_key = first_elem.tuple[0].clone();
    let mut agg_group = vec![DynamicElement::new(first_elem.tuple[1].clone(), first_elem.tag.clone())];

    // Enter main loop
    while let Some(curr_elem) = batch.next_elem() {
      let curr_key = curr_elem.tuple[0].clone();
      let elem = DynamicElement::new(curr_elem.tuple[1].clone(), curr_elem.tag.clone());
      if curr_key == agg_key {
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
    DynamicBatches::single(ElementsBatch::new(result))
  }
}
