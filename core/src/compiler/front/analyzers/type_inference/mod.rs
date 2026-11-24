//! # Type inference analysis

mod analysis;
mod error;
mod foreign_aggregate;
mod foreign_function;
mod foreign_predicate;
mod local;
mod operator_rules;
mod type_set;
mod unification;

use super::super::utils::*;
use super::super::FrontCompileErrorMessage;

pub use analysis::*;
pub use error::*;
pub use foreign_aggregate::*;
pub use foreign_function::*;
pub use foreign_predicate::*;
pub use local::*;
pub use operator_rules::*;
pub use type_set::*;
pub use unification::*;

type Error = FrontCompileErrorMessage;
