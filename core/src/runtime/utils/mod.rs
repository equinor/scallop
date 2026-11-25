mod gallop;
mod join_product;

use std::cmp::Ordering;

use crate::runtime::provenance::{Provenance, Tagged};

pub(crate) use gallop::*;
pub(crate) use join_product::JoinProduct;

pub(crate) fn merge_sorted_tagged<T: Clone + Ord, P: Provenance>(
  mut left: Vec<Tagged<T, P>>,
  mut right: Vec<Tagged<T, P>>,
  ctx: &P,
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

  let mut merged = Vec::with_capacity(left.len() + right.len());
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
