//! Beam Search Scheduling
use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::*;

use super::utils::compute_stable_retain;

/// Invariant: waitlist is sorted by weight from small to large
pub fn schedule_beam<'a, Prov: Provenance>(
  to_add: &'a mut DynamicSortedCollection<Prov>,
  waitlist: &'a mut Vec<DynamicElement<Prov>>,
  stable: &'a mut [DynamicSortedCollection<Prov>],
  ctx: &'a Prov,
  beam_size: usize,
) {
  // First go over the to_add set
  if !to_add.is_empty() {
    let mut to_remove_to_add_indices = HashSet::new();
    for stable_batch in stable.iter_mut() {
      let mut index = 0;

      // Go over the stable batch and retain the related elements
      if to_add.len() > 4 * stable_batch.len() {
        stable_batch.elements.retain_mut(|stable_elem| {
          index = gallop_index(to_add, index, |y| y < stable_elem);
          compute_stable_retain(index, to_add, stable_elem, &mut to_remove_to_add_indices, ctx)
        });
      } else {
        stable_batch.elements.retain_mut(|stable_elem| {
          while index < to_add.len() && &to_add[index] < stable_elem {
            index += 1;
          }
          compute_stable_retain(index, to_add, stable_elem, &mut to_remove_to_add_indices, ctx)
        });
      }
    }

    // Remove the elements in `to_add` that we deem not needed
    to_add.retain(|i, _| !to_remove_to_add_indices.contains(&i));
  }

  if to_add.len() > beam_size {
    let mut sorted_delta = rank_delta(to_add, ctx);
    let split = sorted_delta.len() - beam_size;
    to_add
      .elements
      .extend(sorted_delta.drain(split..).map(|(_, elem)| elem));
    to_add.elements.sort_unstable();

    // Keep older waitlist entries before this delta's overflow.
    waitlist.extend(sorted_delta.into_iter().map(|(_, elem)| elem));
  } else {
    // The delta already fits and is sorted by tuple.
    while to_add.len() < beam_size && !waitlist.is_empty() {
      to_add.insert(waitlist.pop().unwrap(), ctx);
    }
  }

  fn rank_delta<Prov: Provenance>(
    to_add: &mut DynamicSortedCollection<Prov>,
    ctx: &Prov,
  ) -> Vec<(f64, DynamicElement<Prov>)> {
    let mut weighted: Vec<_> = to_add.drain().map(|elem| (ctx.weight(&elem.tag), elem)).collect();
    if weighted.iter().all(|(weight, _)| !weight.is_nan()) {
      // Equal weights were inserted before earlier arrivals, so reverse them before the stable sort.
      weighted.reverse();
      weighted.sort_by(|(left, _), (right, _)| left.partial_cmp(right).unwrap());
      weighted
    } else {
      // Keep the existing comparison and insertion behavior for unordered priorities.
      let mut ranked = Vec::with_capacity(weighted.len());
      for elem in weighted {
        let index = gallop_index(&ranked, 0, |(weight, _)| *weight < elem.0);
        ranked.insert(index, elem);
      }
      ranked
    }
  }
}
