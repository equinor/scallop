use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::gallop_index;

pub(super) fn reconcile_delta<Prov: Provenance>(
  delta: &mut DynamicSortedCollection<Prov>,
  stable: &mut [DynamicSortedCollection<Prov>],
  ctx: &Prov,
) {
  let delta_len = delta.len();
  if delta_len == 0 {
    return;
  }

  let mut saturated = Vec::new();
  let mut remove_from_batch = Vec::new();
  // Defer removing saturated delta rows: every batch must still see them in the original order.
  let mut mark_saturated = |index| {
    if saturated.is_empty() {
      saturated.resize(delta_len, false);
    }
    saturated[index] = true;
  };
  for batch in stable {
    let rows = &mut batch.elements;
    let (Some(first), Some(last)) = (rows.first(), rows.last()) else {
      continue;
    };
    if last < &delta[0] || &delta[delta_len - 1] < first {
      continue;
    }
    let mut index = gallop_index(delta, 0, |row| row < first);

    if rows.len() / 4 >= delta_len {
      let mut stable_index = 0;
      for (delta_index, candidate) in delta.elements.iter_mut().enumerate().skip(index) {
        stable_index = gallop_index(rows, stable_index, |row| row < candidate);
        let Some(old) = rows.get_mut(stable_index) else {
          break;
        };
        if old.tuple == candidate.tuple {
          if compute_saturation_and_update_tag(candidate, old, ctx) {
            mark_saturated(delta_index);
          } else {
            remove_from_batch.push(stable_index);
          }
          stable_index += 1;
        }
      }

      match remove_from_batch.as_slice() {
        [] => {}
        &[only] => {
          rows.remove(only);
        }
        &[first, .., last] => {
          let mut removals = remove_from_batch.iter().copied().peekable();
          let mut index = first;
          rows
            .extract_if(first..=last, |_| {
              let remove = removals.next_if_eq(&index).is_some();
              index += 1;
              remove
            })
            .for_each(drop);
        }
      }
      remove_from_batch.clear();
    } else {
      let seek_delta = delta_len > rows.len().saturating_mul(4);
      rows.retain_mut(|old| {
        if seek_delta {
          index = gallop_index(delta, index, |row| row < old);
        } else {
          while index < delta_len && &delta[index] < old {
            index += 1;
          }
        }
        if index == delta_len || delta[index].tuple != old.tuple {
          return true;
        }
        if compute_saturation_and_update_tag(&mut delta[index], old, ctx) {
          mark_saturated(index);
          true
        } else {
          false
        }
      });
    }
  }
  if !saturated.is_empty() {
    delta.retain(|index, _| !saturated[index]);
  }
}

pub(super) fn schedule_waitlist_row<Prov: Provenance>(
  delta: &mut DynamicSortedCollection<Prov>,
  waitlist: &mut Vec<DynamicElement<Prov>>,
  stable: &mut [DynamicSortedCollection<Prov>],
  ctx: &Prov,
) {
  'next: while let Some(mut pending) = waitlist.pop() {
    for batch in stable.iter_mut() {
      let index = gallop_index(batch, 0, |row| row < &pending);
      if let Some(old) = batch.elements.get_mut(index)
        && old.tuple == pending.tuple
      {
        if compute_saturation_and_update_tag(&mut pending, old, ctx) {
          continue 'next;
        }
        batch.elements.remove(index);
        break;
      }
    }
    delta.insert_unchecked(pending);
    return;
  }
}

/// Compute whether the tag is saturated given a new and old version of the same element.
///
/// Note: Assume that the two elements are of the same tuple
fn compute_saturation_and_update_tag<Prov: Provenance>(
  to_add_elem: &mut DynamicElement<Prov>,
  stable_elem: &mut DynamicElement<Prov>,
  ctx: &Prov,
) -> bool {
  // If the two elements are equal, then we need to compute a new tag, while deciding where
  // to put the new element: stable or recent
  let new_tag = ctx.add(&stable_elem.tag, &to_add_elem.tag);
  let saturated = ctx.saturated(&stable_elem.tag, &new_tag);
  if saturated {
    // If we put the new element in stable, then we retain the stable element and update
    // the tag of that element. Additionally, we will remove the element in the `to_add`
    // collection.
    stable_elem.tag = new_tag;
    true
  } else {
    // If we put the new element in recent, then we will not retain the stable element,
    // while updating tag of the element in `to_add`
    to_add_elem.tag = new_tag;
    false
  }
}
