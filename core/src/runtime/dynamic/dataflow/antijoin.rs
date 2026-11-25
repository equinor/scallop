use super::*;

/// Note: d2 should always be a DynamicCollection
pub struct DynamicAntijoinDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicAntijoinDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicAntijoinDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::binary_recent(&self.d1, &self.d2, AntijoinOp { ctx: self.ctx })
  }
}

pub struct AntijoinOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for AntijoinOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for AntijoinOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, mut i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next_elem();
    let i2_curr = i2.next_elem();
    DynamicBatch::new(DynamicAntijoinBatch {
      i1,
      i1_curr,
      i2,
      i2_curr,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicAntijoinBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2: DynamicBatch<'a, Prov>,
  i2_curr: Option<DynamicElement<Prov>>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicAntijoinBatch<'a, Prov> {
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

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicAntijoinBatch<'a, Prov> {
  fn is_tuple_sorted(&self) -> bool {
    self.i1.is_tuple_sorted()
  }

  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    while let Some(left) = &mut self.i1_curr {
      if let Some(right) = &self.i2_curr {
        match left.tuple[0].cmp(&right.tuple) {
          Ordering::Less => {}
          Ordering::Equal => {
            if let Some(tag) = self.ctx.minus(&left.tag, &right.tag) {
              left.tag = tag;
            } else {
              self.i1_curr = self.i1.next_elem();
              continue;
            }
          }
          Ordering::Greater => {
            self.i2_curr = self.i2.search_until(&left.tuple[0]);
            continue;
          }
        }
      }
      return std::mem::replace(&mut self.i1_curr, self.i1.next_elem());
    }
    None
  }
}
