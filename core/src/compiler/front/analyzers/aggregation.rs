use super::super::ast::*;
use super::super::error::*;
use super::super::source::*;
use super::super::utils::*;

#[derive(Debug, Clone)]
pub struct AggregationAnalysis {
  pub errors: Vec<AggregationAnalysisError>,
}

impl AggregationAnalysis {
  pub fn new() -> Self {
    Self { errors: vec![] }
  }
}

impl Default for AggregationAnalysis {
  fn default() -> Self {
    Self::new()
  }
}

impl NodeVisitor<Reduce> for AggregationAnalysis {
  fn visit(&mut self, reduce: &Reduce) {
    // Check max/min arg
    if reduce.operator().name().name().as_str() == "forall" {
      // Check the body of forall expression
      match reduce.body() {
        Formula::Implies(_) => {}
        _ => self.errors.push(AggregationAnalysisError::ForallBodyNotImplies {
          loc: reduce.location().clone(),
        }),
      }
    }
  }
}

#[derive(Debug, Clone)]
pub enum AggregationAnalysisError {
  ForallBodyNotImplies { loc: Loc },
}

impl FrontCompileErrorTrait for AggregationAnalysisError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::ForallBodyNotImplies { loc } => {
        format!(
          "the body of forall aggregation must be an `implies` formula\n{}",
          loc.report(src)
        )
      }
    }
  }
}
