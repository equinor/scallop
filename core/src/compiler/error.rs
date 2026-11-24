use super::back::BackCompileError;
use super::front::FrontCompileError;

#[derive(Clone, Debug, thiserror::Error)]
pub enum CompileError {
  #[error(transparent)]
  Front(FrontCompileError),
  #[error(transparent)]
  Back(BackCompileError),
}

pub type CompileErrors = Vec<CompileError>;
