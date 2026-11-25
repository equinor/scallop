use std::path::PathBuf;

use csv::StringRecord;

use crate::common::input_file::InputFile;
use crate::runtime::error::IOError;

#[derive(Debug, Clone)]
pub enum InputFileContent {
  CSV(CSVFileContent),
}

impl InputFileContent {
  pub fn load(input_file: &InputFile) -> Result<Self, IOError> {
    match input_file {
      InputFile::Csv {
        file_path,
        deliminator,
        has_header,
        ..
      } => CSVFileContent::from_file(file_path.clone(), *deliminator, *has_header).map(InputFileContent::CSV),
    }
  }
}

#[derive(Debug, Clone)]
pub struct CSVFileContent {
  fields: Vec<String>,
  rows: Vec<StringRecord>,
}

impl CSVFileContent {
  pub fn from_file(file_path: PathBuf, deliminator: u8, has_header: bool) -> Result<Self, IOError> {
    let mut rdr = csv::ReaderBuilder::new()
      .delimiter(deliminator)
      .has_headers(has_header)
      .from_path(file_path.clone())
      .map_err(|e| IOError::CannotOpenFile {
        file_path,
        error: format!("{}", e),
      })?;

    // Generate fields
    let fields = if has_header {
      rdr
        .headers()
        .map_err(|e| IOError::CannotReadHeader {
          error: format!("{}", e),
        })?
        .iter()
        .map(|s| s.to_string())
        .collect()
    } else {
      vec![]
    };

    let rows = rdr
      .into_records()
      .map(|record| record.map_err(|e| IOError::CannotParseCSV { error: e.to_string() }))
      .collect::<Result<_, _>>()?;

    Ok(Self { fields, rows })
  }

  pub fn num_columns(&self) -> usize {
    self.fields.len()
  }

  pub fn num_rows(&self) -> usize {
    self.rows.len()
  }

  pub fn get_header_id(&self, header: &str) -> Option<usize> {
    self.fields.iter().rposition(|field| field == header)
  }

  pub fn get_ith_header(&self, i: usize) -> Option<&String> {
    self.fields.get(i)
  }

  pub fn headers(&self) -> &Vec<String> {
    &self.fields
  }

  pub fn iter_headers(&self) -> impl Iterator<Item = &String> {
    self.fields.iter()
  }

  pub fn get_rows(&self) -> impl Iterator<Item = &StringRecord> {
    self.rows.iter()
  }
}
