use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::utils::{reconcile_delta, schedule_waitlist_row};

pub fn schedule_dfs<'a, Prov: Provenance>(
  to_add: &'a mut DynamicSortedCollection<Prov>,
  waitlist: &'a mut Vec<DynamicElement<Prov>>,
  stable: &'a mut [DynamicSortedCollection<Prov>],
  ctx: &'a Prov,
) {
  reconcile_delta(to_add, stable, ctx);

  // If there exists at least one element left in the to_add set
  if !to_add.is_empty() {
    // Then we leave only one element in to_add and leave everything else to waitlist
    let num_elems = to_add.len();
    let mut drainer = to_add.drain();
    let first = drainer.next().unwrap(); // Unwrap will go through since delta is not empty

    // Insert everything into waitlist
    waitlist.reserve(num_elems - 1); // num_elems - 1 is okay since there is at least one elem
    waitlist.extend(drainer);

    // For the first element, we add that back to delta
    to_add.insert_unchecked(first);
  } else {
    schedule_waitlist_row(to_add, waitlist, stable, ctx);
  }
}
