use super::*;
use crate::common::tuple::Tuple;

pub struct DynamicJoinDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicJoinDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicJoinDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let op = JoinOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let op = JoinOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone()),
    ])
  }
}

pub struct JoinOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for JoinOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for JoinOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, mut i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next_elem();
    let i2_curr = i2.next_elem();
    DynamicBatch::new(DynamicJoinBatch {
      i1,
      i1_curr,
      i2,
      i2_curr,
      right_group: Vec::new(),
      right_index: 0,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicJoinBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2: DynamicBatch<'a, Prov>,
  i2_curr: Option<DynamicElement<Prov>>,
  right_group: Vec<DynamicElement<Prov>>,
  right_index: usize,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicJoinBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2: self.i2.clone(),
      i2_curr: self.i2_curr.clone(),
      right_group: self.right_group.clone(),
      right_index: self.right_index,
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicJoinBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    while let Some(left) = &self.i1_curr {
      if let Some(right) = self.right_group.get(self.right_index) {
        self.right_index += 1;
        let tuple = Tuple::from((left.tuple[0].clone(), left.tuple[1].clone(), right.tuple[1].clone()));
        let tag = self.ctx.mult(&left.tag, &right.tag);
        return Some(DynamicElement::new(tuple, tag));
      }

      if let Some(first_right) = self.right_group.first() {
        self.i1_curr = self.i1.next_elem();
        self.right_index = 0;
        if !self
          .i1_curr
          .as_ref()
          .is_some_and(|next| next.tuple[0] == first_right.tuple[0])
        {
          self.right_group.clear();
        }
        continue;
      }

      let right = self.i2_curr.as_ref()?;
      let key = &left.tuple[0];
      match key.cmp(&right.tuple[0]) {
        Ordering::Less => self.i1_curr = self.i1.search_elem_0_until(&right.tuple[0]),
        Ordering::Equal => {
          // Keep the first row of the next key as lookahead; only the right group is replayed.
          self.right_group.push(self.i2_curr.take().unwrap());
          while let Some(next) = self.i2.next_elem() {
            if &next.tuple[0] != key {
              self.i2_curr = Some(next);
              break;
            }
            self.right_group.push(next);
          }
        }
        Ordering::Greater => self.i2_curr = self.i2.search_elem_0_until(key),
      }
    }
    None
  }
}
