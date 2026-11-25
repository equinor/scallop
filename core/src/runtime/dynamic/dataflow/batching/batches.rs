use super::*;
use crate::runtime::provenance::*;

pub trait Batches<'a, Prov: Provenance>: dyn_clone::DynClone + 'a {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>>;
}

dyn_clone::clone_trait_object!(<'a, Prov> Batches<'a, Prov> where Prov: Provenance);

#[derive(Clone)]
pub struct DynamicBatches<'a, Prov: Provenance>(Box<dyn Batches<'a, Prov>>);

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.0.next_batch()
  }
}

impl<'a, Prov: Provenance> DynamicBatches<'a, Prov> {
  pub fn new<B: Batches<'a, Prov>>(b: B) -> Self {
    Self(Box::new(b))
  }

  pub fn from_iterator(iter: impl Iterator<Item = DynamicBatch<'a, Prov>> + Clone + 'a) -> Self {
    Self::new(IteratorBatches(iter))
  }

  pub fn empty() -> Self {
    Self::from_iterator(std::iter::empty())
  }

  pub fn single<B: Batch<'a, Prov>>(b: B) -> Self {
    Self::from_iterator(std::iter::once(DynamicBatch::new(b)))
  }

  pub fn chain(bs: Vec<DynamicBatches<'a, Prov>>) -> Self {
    Self::from_iterator(bs.into_iter().flatten())
  }

  pub fn binary<Op: BatchBinaryOp<'a, Prov>>(self, b2: Self, op: Op) -> Self {
    Self::new(DynamicBatchesBinary::new(self, b2, op))
  }

  pub fn unary<Op: BatchUnaryOp<'a, Prov> + Clone>(self, op: Op) -> Self {
    Self::from_iterator(self.map(move |batch| op.apply(batch)))
  }
}
impl<'a, Prov: Provenance> Iterator for DynamicBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.next_batch()
  }
}

#[derive(Clone)]
struct IteratorBatches<I>(I);

impl<'a, Prov, I> Batches<'a, Prov> for IteratorBatches<I>
where
  Prov: Provenance,
  I: Iterator<Item = DynamicBatch<'a, Prov>> + Clone + 'a,
{
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.0.next()
  }
}

#[derive(Clone)]
pub struct DynamicBatchesBinary<'a, Prov: Provenance> {
  b1: DynamicBatches<'a, Prov>,
  b1_curr: Option<DynamicBatch<'a, Prov>>,
  b2: DynamicBatches<'a, Prov>,
  b2_source: DynamicBatches<'a, Prov>,
  op: DynamicBatchBinaryOp<'a, Prov>,
}

impl<'a, Prov: Provenance> DynamicBatchesBinary<'a, Prov> {
  pub fn new<Op: BatchBinaryOp<'a, Prov>>(
    mut b1: DynamicBatches<'a, Prov>,
    b2: DynamicBatches<'a, Prov>,
    op: Op,
  ) -> Self {
    let b1_curr = b1.next_batch();
    let b2_source = b2.clone();
    Self {
      b1,
      b1_curr,
      b2,
      b2_source,
      op: DynamicBatchBinaryOp::new(op),
    }
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatchesBinary<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    loop {
      let b1_curr = self.b1_curr.as_ref()?;
      if let Some(b2_curr) = self.b2.next_batch() {
        return Some(self.op.apply(b1_curr.clone(), b2_curr));
      }
      self.b1_curr = self.b1.next_batch();
      self.b2 = self.b2_source.clone();
    }
  }
}
