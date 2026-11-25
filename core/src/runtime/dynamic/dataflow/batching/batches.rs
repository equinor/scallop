use super::super::{Dataflow, DynamicDataflow};
use super::*;
use crate::runtime::provenance::*;

pub trait Batches<'a, Prov: Provenance>: dyn_clone::DynClone + 'a {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>>;

  /// As with `Iterator::size_hint`, bounds the number of batches before the next exhaustion.
  fn batch_size_hint(&self) -> (usize, Option<usize>) {
    (0, None)
  }
}

dyn_clone::clone_trait_object!(<'a, Prov> Batches<'a, Prov> where Prov: Provenance);

#[derive(Clone)]
pub struct DynamicBatches<'a, Prov: Provenance>(Box<dyn Batches<'a, Prov>>);

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.0.next_batch()
  }

  fn batch_size_hint(&self) -> (usize, Option<usize>) {
    self.0.batch_size_hint()
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

  pub(crate) fn binary_recent<Op: BatchBinaryOp<'a, Prov> + Clone>(
    left: &DynamicDataflow<'a, Prov>,
    right: &DynamicDataflow<'a, Prov>,
    op: Op,
  ) -> Self {
    let pairs = [
      left.iter_stable().binary(right.iter_recent(), op.clone()),
      left.iter_recent().binary(right.iter_stable(), op.clone()),
      left.iter_recent().binary(right.iter_recent(), op),
    ];
    Self::from_iterator(pairs.into_iter().flatten())
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

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.batch_size_hint()
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

  fn batch_size_hint(&self) -> (usize, Option<usize>) {
    self.0.size_hint()
  }
}

#[derive(Clone)]
pub struct DynamicBatchesBinary<'a, Prov: Provenance> {
  b1: DynamicBatches<'a, Prov>,
  b1_curr: Option<DynamicBatch<'a, Prov>>,
  left_moved: bool,
  b2: DynamicBatches<'a, Prov>,
  right_single: bool,
  b2_source: Option<DynamicBatches<'a, Prov>>,
  op: DynamicBatchBinaryOp<'a, Prov>,
}

impl<'a, Prov: Provenance> DynamicBatchesBinary<'a, Prov> {
  pub fn new<Op: BatchBinaryOp<'a, Prov>>(
    mut b1: DynamicBatches<'a, Prov>,
    b2: DynamicBatches<'a, Prov>,
    op: Op,
  ) -> Self {
    let b1_curr = b1.next_batch();
    let right_single = b1_curr.is_some() && b2.size_hint().1 == Some(1);
    let b2_source = (b1_curr.is_some() && b1.size_hint().1 != Some(0)).then(|| b2.clone());
    Self {
      b1,
      b1_curr,
      left_moved: false,
      b2,
      right_single,
      b2_source,
      op: DynamicBatchBinaryOp::new(op),
    }
  }

  fn next_single_right(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    loop {
      if self.b1_curr.is_none() && !self.left_moved {
        return None;
      }
      if let Some(right) = self.b2.next_batch() {
        let left = self.b1_curr.take().expect("left batch must exist for its only pairing");
        self.left_moved = true;
        return Some(self.op.apply(left, right));
      }
      // Advance the left input only after the right cursor has reported exhaustion.
      self.left_moved = false;
      self.advance_left();
    }
  }

  fn advance_left(&mut self) {
    self.b1_curr = self.b1.next_batch();
    if self.b1_curr.is_some() {
      self.b2 = if self.b1.size_hint().1 == Some(0) {
        self.b2_source.take()
      } else {
        self.b2_source.clone()
      }
      .expect("another left batch requires a right replay cursor");
    }
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatchesBinary<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    if self.right_single {
      return self.next_single_right();
    }
    loop {
      let left = self.b1_curr.as_ref()?;
      if let Some(right) = self.b2.next_batch() {
        return Some(self.op.apply(left.clone(), right));
      }
      self.advance_left();
    }
  }
}

#[cfg(test)]
mod tests {
  use std::cell::Cell;

  use super::*;
  use crate::common::tuple::Tuple;
  use crate::runtime::dynamic::DynamicElement;
  use crate::runtime::provenance::unit::{Unit, UnitProvenance};

  struct CountedBatch<'a> {
    row: Option<i32>,
    clones: &'a Cell<usize>,
  }

  impl Clone for CountedBatch<'_> {
    fn clone(&self) -> Self {
      self.clones.set(self.clones.get() + 1);
      Self {
        row: self.row,
        clones: self.clones,
      }
    }
  }

  impl<'a> Batch<'a, UnitProvenance> for CountedBatch<'a> {
    fn next_elem(&mut self) -> Option<DynamicElement<UnitProvenance>> {
      self.row.take().map(|row| DynamicElement::new(row, Unit))
    }
  }

  #[derive(Clone)]
  struct PairRows;

  impl<'a> BatchBinaryOp<'a, UnitProvenance> for PairRows {
    fn apply(
      &self,
      left: DynamicBatch<'a, UnitProvenance>,
      right: DynamicBatch<'a, UnitProvenance>,
    ) -> DynamicBatch<'a, UnitProvenance> {
      DynamicBatch::from_iterator(
        left
          .zip(right)
          .map(|(left, right)| DynamicElement::new((left.tuple, right.tuple), Unit)),
      )
    }
  }

  #[test]
  fn single_batch_pair_moves_both_batches() {
    let clones = Cell::new(0);
    let single = |row| {
      DynamicBatches::single(CountedBatch {
        row: Some(row),
        clones: &clones,
      })
    };
    let mut pairs = single(1).binary(single(2), PairRows);
    assert_eq!(pairs.next().unwrap().next().unwrap().tuple, Tuple::from((1i32, 2i32)));
    assert!(pairs.next().is_none());
    assert_eq!(clones.get(), 0);
  }

  #[test]
  fn batch_pairing_keeps_left_major_order_without_looking_ahead() {
    for (known, right_len) in [(true, 1), (false, 1), (true, 2), (false, 2)] {
      let clones = Cell::new(0);
      let left_pulls = Cell::new(0);
      let right_pulls = Cell::new(0);
      let make = |row| {
        DynamicBatch::new(CountedBatch {
          row: Some(row),
          clones: &clones,
        })
      };
      let left = DynamicBatches::from_iterator([1, 2].into_iter().map(|row| {
        left_pulls.set(left_pulls.get() + 1);
        make(row)
      }));
      let mut right = (3..3 + right_len).map(|row| {
        right_pulls.set(right_pulls.get() + 1);
        make(row)
      });
      let right = if known {
        DynamicBatches::from_iterator(right)
      } else {
        DynamicBatches::from_iterator(std::iter::from_fn(move || right.next()))
      };
      let mut pairs = left.binary(right, PairRows);
      assert_eq!((left_pulls.get(), right_pulls.get()), (1, 0));
      for (pair, (left, right)) in (1..=2)
        .flat_map(|left| (3..3 + right_len).map(move |right| (left, right)))
        .enumerate()
      {
        assert_eq!(pairs.next().unwrap().next().unwrap().tuple, Tuple::from((left, right)));
        assert_eq!((left_pulls.get(), right_pulls.get()), (left as usize, pair + 1));
      }
      assert!(pairs.next().is_none());
      assert_eq!(
        clones.get(),
        if known && right_len == 1 {
          0
        } else {
          2 * right_len as usize
        }
      );
    }
  }
}
