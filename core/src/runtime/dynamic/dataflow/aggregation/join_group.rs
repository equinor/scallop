use super::*;

pub struct DynamicAggregationJoinGroupDataflow<'a, Prov: Provenance> {
  pub agg: DynamicAggregator<Prov>,
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Clone for DynamicAggregationJoinGroupDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
      runtime: self.runtime,
    }
  }
}

impl<'a, Prov: Provenance> DynamicAggregationJoinGroupDataflow<'a, Prov> {
  pub fn new(
    agg: DynamicAggregator<Prov>,
    d1: DynamicDataflow<'a, Prov>,
    d2: DynamicDataflow<'a, Prov>,
    ctx: &'a Prov,
    runtime: &'a RuntimeEnvironment,
  ) -> Self {
    Self {
      agg,
      d1,
      d2,
      ctx,
      runtime,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicAggregationJoinGroupDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let Some(group_by) = self.d1.iter_recent().next_batch() else {
      return DynamicBatches::empty();
    };

    let Some(mut main) = self.d2.iter_recent().next_batch() else {
      return DynamicBatches::empty();
    };

    let mut group_by = group_by.peekable();
    let seekable = main.is_materialized_and_seekable();
    let mut current: Option<DynamicElement<Prov>> = None;
    let mut result = Vec::new();
    while let Some(first) = group_by.next() {
      let key = &first.tuple[0];
      if current.as_ref().is_none_or(|row| &row.tuple[0] < key) {
        current = if seekable {
          main.search_elem_0_until(key)
        } else {
          main.find(|row| &row.tuple[0] >= key)
        };
      }

      let inputs: DynamicElements<Prov> = std::iter::from_fn(|| {
        let row = current.take().or_else(|| main.next_elem())?;
        if &row.tuple[0] == key {
          Some(row)
        } else {
          current = Some(row);
          None
        }
      })
      .map(|row| DynamicElement::new(row.tuple.drain(1).expect("missing aggregate input payload"), row.tag))
      .collect();
      let matched = !inputs.is_empty();
      let aggregates = self.agg.aggregate(self.ctx, self.runtime, inputs);
      let mut append = |row: &DynamicElement<Prov>| {
        let payload = &row.tuple[1];
        result.extend(aggregates.iter().map(|aggregate| {
          DynamicElement::new(
            (key.clone(), payload.clone(), aggregate.tuple.clone()),
            self.ctx.mult(&row.tag, &aggregate.tag),
          )
        }));
      };

      append(&first);
      // Unmatched rows each get their own empty-group aggregate.
      if matched {
        while let Some(row) = group_by.next_if(|row| &row.tuple[0] == key) {
          append(&row);
        }
      }
    }

    DynamicBatches::single(ElementsBatch::new(result))
  }
}
