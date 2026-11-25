use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::utils::reconcile_delta;

pub fn schedule_lfp<'a, Prov: Provenance>(
  to_add: &'a mut DynamicSortedCollection<Prov>,
  _waitlist: &'a mut [DynamicElement<Prov>],
  stable: &'a mut [DynamicSortedCollection<Prov>],
  ctx: &'a Prov,
) {
  reconcile_delta(to_add, stable, ctx);
}
