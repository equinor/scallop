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

fn chosen_set_tag<P: Provenance>(prov: &P, batch: &[DynamicElement<P>], chosen: &[usize]) -> Option<P::Tag> {
  batch.iter().enumerate().try_fold(prov.one(), |acc, (i, elem)| {
    if chosen.contains(&i) {
      Some(prov.mult(&acc, &elem.tag))
    } else {
      let negated = prov.negate(&elem.tag)?;
      Some(prov.mult(&acc, &negated))
    }
  })
}
