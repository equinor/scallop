use super::super::*;

#[derive(Clone, Debug, thiserror::Error)]
pub enum AttributeError {
  #[error("Duplicated attribute definitions for `@{name}`")]
  DuplicatedAttributeProcessor { name: String },
  #[error("Unknown attribute `@{name}`")]
  UnknownAttribute { name: String },
  #[error("Attribute `@{name}` is reserved in Scallop")]
  ReservedAttribute { name: String },
  #[error("{msg}")]
  Custom { msg: String },
}

impl AttributeError {
  pub fn new_custom(msg: String) -> Self {
    Self::Custom { msg }
  }
}

impl FrontCompileErrorTrait for AttributeError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, _: &Sources) -> String {
    format!("{}", self)
  }
}
