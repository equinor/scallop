use super::*;

#[derive(Debug, Clone, thiserror::Error)]
pub enum BackCompileError {
  #[error(transparent)]
  SCCError(#[from] SCCError),
  #[error(transparent)]
  DemandTransformError(#[from] optimizations::DemandTransformError),
}
