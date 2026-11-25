use super::*;

pub struct DynamicDifferenceDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicDifferenceDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicDifferenceDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::binary_recent(&self.d1, &self.d2, DifferenceOp { ctx: self.ctx })
  }
}

pub struct DifferenceOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DifferenceOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for DifferenceOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, mut i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next_elem();
    let i2_curr = i2.next_elem();
    DynamicBatch::new(DynamicDifferenceBatch {
      i1,
      i1_curr,
      i2,
      i2_curr,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicDifferenceBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2: DynamicBatch<'a, Prov>,
  i2_curr: Option<DynamicElement<Prov>>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicDifferenceBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2: self.i2.clone(),
      i2_curr: self.i2_curr.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicDifferenceBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    loop {
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr_elem), Some(i2_curr_elem)) => match i1_curr_elem.tuple.cmp(&i2_curr_elem.tuple) {
          Ordering::Less => {
            return std::mem::replace(&mut self.i1_curr, self.i1.next_elem());
          }
          Ordering::Equal => {
            let maybe_tag = self.ctx.minus(&i1_curr_elem.tag, &i2_curr_elem.tag);
            let result = std::mem::replace(&mut self.i1_curr, self.i1.next_elem());
            self.i2_curr = self.i2.next_elem();
            if let Some(tag) = maybe_tag {
              return result.map(|mut elem| {
                elem.tag = tag;
                elem
              });
            }
          }
          Ordering::Greater => {
            self.i2_curr = self.i2.search_until(&i1_curr_elem.tuple);
          }
        },
        (Some(_), None) => return std::mem::replace(&mut self.i1_curr, self.i1.next_elem()),
        _ => return None,
      }
    }
  }
}
