//! A-Star Search Scheduling
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::gallop_index;

use super::utils::{reconcile_delta, schedule_waitlist_row};

/// Assumption: waitlist is sorted by weight from small to large
pub fn schedule_a_star<'a, Prov: Provenance>(
  to_add: &'a mut DynamicSortedCollection<Prov>,
  waitlist: &'a mut Vec<DynamicElement<Prov>>,
  stable: &'a mut [DynamicSortedCollection<Prov>],
  ctx: &'a Prov,
) {
  reconcile_delta(to_add, stable, ctx);

  // Drain to_add and merge it into waitlist. This preserves the order of elements in the waitlist
  for to_add_elem in to_add.drain() {
    // Store the weight
    let to_add_elem_weight = ctx.weight(&to_add_elem.tag);

    // Check if we can insert to_add_element in the middle of waitlist
    let to_insert_index = gallop_index(waitlist, 0, |elem| ctx.weight(&elem.tag) < to_add_elem_weight);
    if let Some(existing_elem) = waitlist.get_mut(to_insert_index) {
      if existing_elem.tuple == to_add_elem.tuple {
        existing_elem.tag = ctx.add(&existing_elem.tag, &to_add_elem.tag);
      } else {
        waitlist.insert(to_insert_index, to_add_elem);
      }
    } else {
      waitlist.push(to_add_elem);
    }
  }

  schedule_waitlist_row(to_add, waitlist, stable, ctx);
}
