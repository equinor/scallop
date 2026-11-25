use super::io::IOError;
use crate::common::foreign_function::ForeignFunctionError;
use crate::common::foreign_predicate::ForeignPredicateError;
use crate::compiler::front::attribute::AttributeError;
use crate::runtime::database::DatabaseError;

/// A runtime failure whose underlying error is available through [`std::error::Error::source`].
#[derive(Clone, Debug, thiserror::Error)]
pub enum RuntimeError {
  #[error("{0}")]
  IO(#[from] IOError),
  #[error("{0}")]
  ForeignFunction(#[source] ForeignFunctionError),
  #[error("{0}")]
  ForeignPredicate(#[source] ForeignPredicateError),
  #[error("{0}")]
  ForeignAttribute(#[source] AttributeError),
  #[error("{0}")]
  Database(#[from] DatabaseError),
}
