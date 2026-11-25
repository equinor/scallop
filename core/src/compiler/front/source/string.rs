use super::*;

#[derive(Clone, Debug)]
pub struct StringSource {
  pub string: String,
  pub line_offset_len: Vec<(usize, usize)>,
}

impl StringSource {
  pub fn new(string: String) -> Self {
    let line_offset_len = collect_line_offset_length(&string);
    Self {
      string,
      line_offset_len,
    }
  }
}

impl Source for StringSource {
  fn content(&self) -> &str {
    &self.string
  }

  fn name(&self) -> Option<&str> {
    None
  }

  fn num_rows(&self) -> usize {
    self.line_offset_len.len()
  }

  fn row_offset_length(&self, row: usize) -> (usize, usize) {
    self.line_offset_len[row]
  }
}
