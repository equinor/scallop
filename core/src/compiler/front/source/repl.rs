use std::borrow::Cow;

use super::*;

#[derive(Clone, Debug)]
pub struct ReplSource {
  pub id: usize,
  pub content: String,
}

impl ReplSource {
  pub fn new(id: usize, content: String) -> Self {
    Self { id, content }
  }
}

impl Source for ReplSource {
  fn content(&self) -> &str {
    &self.content
  }

  fn name(&self) -> Option<&str> {
    None
  }

  fn diagnostic_name(&self) -> Option<Cow<'_, str>> {
    Some(format!("REPL:{}", self.id).into())
  }

  fn num_rows(&self) -> usize {
    1
  }

  fn row_offset_length(&self, _: usize) -> (usize, usize) {
    (0, self.content.len())
  }
}
