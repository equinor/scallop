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
    let Some(batch) = self.d.iter_recent().next_batch() else {
      return DynamicBatches::empty();
    };
    let mut rows = batch
      .map(|row| {
        let Tuple::Tuple(fields) = row.tuple else {
          panic!("missing aggregate input key and payload");
        };
        let mut fields = fields.into_vec().into_iter();
        let key = fields.next().expect("missing aggregate input key");
        let payload = fields.next().expect("missing aggregate input payload");
        (key, DynamicElement::new(payload, row.tag))
      })
      .peekable();
    if rows.peek().is_none() {
      return DynamicBatches::empty();
    }

    let mut result = Vec::new();
    while let Some((key, first)) = rows.next() {
      let group = std::iter::once(first)
        .chain(std::iter::from_fn(|| rows.next_if(|(next_key, _)| next_key == &key)).map(|(_, row)| row))
        .collect();
      let aggregates = self.agg.aggregate(self.ctx, self.runtime, group);
      result.extend(
        aggregates
          .into_iter()
          .map(|row| DynamicElement::new((key.clone(), row.tuple), row.tag)),
      );
    }

    DynamicBatches::single(ElementsBatch::new(result))
  }
}
