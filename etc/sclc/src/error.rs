#[derive(Debug, thiserror::Error)]
pub enum SclcError {
  #[error(transparent)]
  RustEmission(#[from] scallop_core::compiler::ram::RustEmissionError),
  #[error(transparent)]
  Io(#[from] std::io::Error),
  #[error("set SCALLOPDIR to the root of the Scallop source directory before using sclc")]
  MissingScallopDir,
  #[error("{0} failed:\n{1}")]
  Build(&'static str, String),
  #[error("Unknown provenance `{0}`")]
  UnknownProvenance(String),
}
