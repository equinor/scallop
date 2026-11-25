mod gallop;

use std::cmp::Ordering;

use crate::runtime::provenance::{Provenance, Tagged};

pub(crate) use gallop::*;

pub(crate) fn merge_sorted_tagged<T: Clone + Ord, P: Provenance>(
  mut left: Vec<Tagged<T, P>>,
  mut right: Vec<Tagged<T, P>>,
  ctx: &P,
  scratch: Option<&mut Vec<Tagged<T, P>>>,
) -> Vec<Tagged<T, P>> {
  let (Some(first_left), Some(first_right)) = (left.first(), right.first()) else {
    return if left.is_empty() { right } else { left };
  };
  if first_left > first_right {
    std::mem::swap(&mut left, &mut right);
  }
  if left.last() < right.first() {
    left.extend(right);
    return left;
  }

  let capacity = left.len() + right.len();
  if let Some(scratch) = scratch {
    let mut merged = std::mem::take(scratch);
    merged.reserve(capacity);
    // Keep an emptied input allocation for the next pair without changing merge order.
    if left.capacity() >= right.capacity() {
      let merged = merge_into(merged, left.drain(..), right, ctx);
      *scratch = left;
      merged
    } else {
      let merged = merge_into(merged, left, right.drain(..), ctx);
      *scratch = right;
      merged
    }
  } else {
    merge_into(Vec::with_capacity(capacity), left, right, ctx)
  }
}

#[inline]
fn merge_into<T: Clone + Ord, P: Provenance>(
  mut merged: Vec<Tagged<T, P>>,
  left: impl IntoIterator<Item = Tagged<T, P>>,
  right: impl IntoIterator<Item = Tagged<T, P>>,
  ctx: &P,
) -> Vec<Tagged<T, P>> {
  let mut right = right.into_iter().peekable();
  for mut elem in left {
    while let Some(other) = right.peek() {
      match other.cmp(&elem) {
        Ordering::Less => merged.push(right.next().unwrap()),
        Ordering::Equal => {
          elem.tag = ctx.add(&elem.tag, &other.tag);
          right.next();
          break;
        }
        Ordering::Greater => break,
      }
    }
    merged.push(elem);
  }
  merged.extend(right);
  merged
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::runtime::provenance::diff_max_mult_prob::DiffMaxMultProbProvenance;
  use crate::runtime::provenance::DualNumber2;
  use crate::utils::RcFamily;

  #[test]
  fn consecutive_overlapping_merges_reuse_storage_and_keep_tag_order() {
    type Prov = DiffMaxMultProbProvenance<(), RcFamily>;
    let ctx = Prov::default();
    let row = |tuple, id| Tagged::<i32, Prov> {
      tuple,
      tag: DualNumber2::new(id, 0.5),
    };
    let mut left = Vec::with_capacity(16);
    left.extend([row(2, 0), row(3, 1), row(6, 2)]);
    let recycled = left.as_ptr();
    let mut buffer = Vec::new();
    let snapshot = |rows: &[Tagged<i32, Prov>]| {
      rows
        .iter()
        .map(|row| (row.tuple, row.tag.gradient.indices[0]))
        .collect::<Vec<_>>()
    };

    let merged = merge_sorted_tagged(left, vec![row(1, 3), row(3, 4)], &ctx, Some(&mut buffer));
    assert_eq!(snapshot(&merged), [(1, 3), (2, 0), (3, 1), (6, 2)]);
    let merged = merge_sorted_tagged(merged, vec![row(1, 5), row(3, 6), row(7, 7)], &ctx, Some(&mut buffer));
    assert_eq!(snapshot(&merged), [(1, 5), (2, 0), (3, 6), (6, 2), (7, 7)]);
    assert_eq!(merged.as_ptr(), recycled);
  }
}
