use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::runtime::error::*;

#[derive(Clone, Debug, thiserror::Error)]
pub enum DatabaseError {
  #[error("Type mismatch on tuple `{tuple}` for relation `{relation}`. Expected tuple type `{relation_type}`")]
  TypeError {
    relation: String,
    relation_type: TupleType,
    tuple: Tuple,
  },
  #[error("Unknown relation `{relation}`")]
  UnknownRelation { relation: String },
  #[error("New facts in program declared for relation `{relation}`; cannot incrementally compute")]
  NewProgramFacts { relation: String },
  #[error("{0}")]
  IO(#[source] IOError),
}
