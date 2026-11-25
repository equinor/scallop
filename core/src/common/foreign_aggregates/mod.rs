mod avg;
mod categorical;
mod count;
mod disjunct;
mod enumerate;
mod exists;
mod min_max;
mod normalize;
mod rank;
mod sampler;
mod sort;
mod string_join;
mod sum_prod;
mod top_k;
mod uniform;
mod weighted_sum_avg;

pub use avg::*;
pub use categorical::*;
pub use count::*;
pub use disjunct::*;
pub use enumerate::*;
pub use exists::*;
pub use min_max::*;
pub use normalize::*;
pub use rank::*;
pub use sampler::*;
pub use sort::*;
pub use string_join::*;
pub use sum_prod::*;
pub use top_k::*;
pub use uniform::*;
pub use weighted_sum_avg::*;

use crate::runtime::dynamic::DynamicElement;
use crate::runtime::provenance::Provenance;

use super::foreign_aggregate::*;

// The powerset of 0..batch.len() produces each subset's indices in increasing order.
fn chosen_set_tag<P: Provenance>(prov: &P, batch: &[DynamicElement<P>], mut chosen: &[usize]) -> Option<P::Tag> {
  batch.iter().enumerate().try_fold(prov.one(), |acc, (i, elem)| {
    if chosen.first() == Some(&i) {
      chosen = &chosen[1..];
      Some(prov.mult(&acc, &elem.tag))
    } else {
      let negated = prov.negate(&elem.tag)?;
      Some(prov.mult(&acc, &negated))
    }
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::runtime::provenance::{add_mult_prob::AddMultProbProvenance, unit::UnitProvenance};

  #[test]
  fn chosen_set_tag_folds_gaps_and_keeps_multiplication_order() {
    let prob = AddMultProbProvenance::default();
    let rows = [0.2, 0.3, 0.4, 0.7].map(|tag| DynamicElement::new((), tag));
    assert_eq!(
      chosen_set_tag(&prob, &rows, &[1, 3]),
      Some((1.0 - 0.2) * 0.3 * (1.0 - 0.4) * 0.7)
    );
    assert_eq!(
      chosen_set_tag(&prob, &rows, &[]),
      Some((1.0 - 0.2) * (1.0 - 0.3) * (1.0 - 0.4) * (1.0 - 0.7))
    );
    let unit = UnitProvenance;
    let rows = [DynamicElement::new((), unit.one())];
    assert!(chosen_set_tag(&unit, &rows, &[]).is_none());
    assert!(chosen_set_tag(&unit, &rows, &[0]).is_some());
  }
}
