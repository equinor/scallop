use std::path::PathBuf;

use crate::common::tuple_type::*;
use crate::common::value_type::*;

#[derive(Clone, Debug, thiserror::Error)]
pub enum IOError {
  #[error("IO: Cannot open file `{file_path}`: {error}")]
  CannotOpenFile { file_path: PathBuf, error: String },
  #[error("IO: Cannot read file: {error}")]
  CannotReadFile { error: String },
  #[error("IO: Cannot parse CSV: {error}")]
  CannotParseCSV { error: String },
  #[error("IO: Cannot read CSV header: {error}")]
  CannotReadHeader { error: String },
  #[error("IO: Cannot find field `{field}`")]
  CannotFindField { field: String },
  #[error("IO: Index out of bounds: {index}")]
  IndexOutOfBounds { index: usize },
  #[error("IO: Invalid tuple type: `{types}`")]
  InvalidType { types: TupleType },
  #[error("IO: Expect `Symbol` type for field; found `{actual}`")]
  ExpectSymbolType { actual: ValueType },
  #[error("IO: Expect `String` type for value; found `{actual}`")]
  ExpectStringType { actual: ValueType },
  #[error("{error}")]
  ValueParseError {
    #[source]
    error: ValueParseError,
  },
  #[error("IO: Cannot parse probability `{value}`")]
  CannotParseProbability { value: String },
  #[error("IO: Arity mismatch; expected {expected}, found {found}")]
  ArityMismatch { expected: usize, found: usize },
  #[error("IO: Cannot write record: {error}")]
  CannotWriteRecord { error: String },
  #[error("IO: Invalid file format")]
  InvalidFileFormat {},
}
