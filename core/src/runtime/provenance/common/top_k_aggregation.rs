use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::utils::FloatKey;

pub(super) struct ByPriority<T> {
  priority: Reverse<FloatKey<f64>>,
  pub(super) value: T,
}

impl<T> ByPriority<T> {
  pub(super) fn max(weight: f64, value: T) -> Self {
    // Negate before reversing the key so NaN has the lowest priority, not the highest.
    Self {
      priority: Reverse(FloatKey(-weight)),
      value,
    }
  }
}

impl<T> PartialEq for ByPriority<T> {
  fn eq(&self, other: &Self) -> bool {
    self.priority == other.priority
  }
}

impl<T> Eq for ByPriority<T> {}

impl<T> PartialOrd for ByPriority<T> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<T> Ord for ByPriority<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.priority.cmp(&other.priority)
  }
}

pub fn aggregate_top_k_helper<F>(num_elements: usize, k: usize, weight_fn: F) -> Vec<usize>
where
  F: Fn(usize) -> f64,
{
  let size = k.min(num_elements);
  if size == 0 {
    return vec![];
  }

  // Keep the weakest retained candidate at the root of the heap.
  let mut heap = BinaryHeap::with_capacity(size);
  for id in 0..size {
    heap.push(Reverse(ByPriority::max(weight_fn(id), id)));
  }
  for id in size..num_elements {
    let elem = Reverse(ByPriority::max(weight_fn(id), id));
    if &elem < heap.peek().unwrap() {
      heap.pop();
      heap.push(elem);
    }
  }

  heap.into_iter().map(|Reverse(elem)| elem.value).collect()
}

pub fn unweighted_aggregate_top_k_helper<T>(elements: Vec<T>, k: usize) -> Vec<T> {
  if elements.len() <= k {
    elements
  } else {
    elements.into_iter().take(k).collect()
  }
}
