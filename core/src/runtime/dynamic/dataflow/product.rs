use super::*;
use crate::common::tuple::Tuple;

pub struct DynamicProductDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicProductDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicProductDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::binary_recent(&self.d1, &self.d2, ProductOp { ctx: self.ctx })
  }
}

pub struct ProductOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for ProductOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for ProductOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let rows = i2.remaining_slice();
    if matches!(rows, Some([])) {
      return DynamicBatch::from_iterator(std::iter::empty());
    }
    let i1_curr = i1.next_elem();
    if i1_curr.is_none() {
      return DynamicBatch::from_iterator(std::iter::empty());
    }
    let right = if rows.is_some() {
      ProductRight::Borrowed { batch: i2, index: 0 }
    } else {
      ProductRight::Streaming {
        start: i2.clone(),
        cursor: i2,
      }
    };
    DynamicBatch::new(DynamicProductBatch {
      i1,
      i1_curr,
      right,
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
enum ProductRight<'a, Prov: Provenance> {
  Borrowed {
    batch: DynamicBatch<'a, Prov>,
    index: usize,
  },
  Streaming {
    start: DynamicBatch<'a, Prov>,
    cursor: DynamicBatch<'a, Prov>,
  },
}

pub struct DynamicProductBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  right: ProductRight<'a, Prov>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicProductBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      right: self.right.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicProductBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(left) = &self.i1_curr {
      let right = match &mut self.right {
        ProductRight::Borrowed { batch, index } => {
          let rows = batch
            .remaining_slice()
            .expect("borrowed product needs contiguous input");
          if let Some(row) = rows.get(*index) {
            *index += 1;
            Some((row.tuple.clone(), self.ctx.mult(&left.tag, &row.tag)))
          } else {
            None
          }
        }
        ProductRight::Streaming { cursor, .. } => cursor
          .next_elem()
          .map(|row| (row.tuple, self.ctx.mult(&left.tag, &row.tag))),
      };
      if let Some((tuple, tag)) = right {
        return Some(DynamicElement::new(Tuple::from((left.tuple.clone(), tuple)), tag));
      }

      self.i1_curr = self.i1.next_elem();
      if self.i1_curr.is_some() {
        match &mut self.right {
          ProductRight::Borrowed { index, .. } => *index = 0,
          ProductRight::Streaming { start, cursor } => *cursor = start.clone(),
        }
      }
    }
    None
  }
}
