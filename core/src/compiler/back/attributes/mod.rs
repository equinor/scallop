mod aggregate;
mod demand;
mod goal;
mod input_file;
mod magic_set;
mod no_temp_relation;
mod scheduler;
mod storage;

pub use aggregate::*;
pub use demand::*;
pub use goal::*;
pub use input_file::*;
pub use magic_set::*;
pub use no_temp_relation::*;
pub use scheduler::*;
pub use storage::*;

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DisjunctiveHeadOrigin {
  pub rule: usize,
  pub variable_positions: Vec<usize>,
}

impl AttributeTrait for DisjunctiveHeadOrigin {
  fn name(&self) -> String {
    "disjunctive_head_origin".to_string()
  }
}
