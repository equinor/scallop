use super::*;

pub(super) type ReduceGroup = (Vec<VariableBinding>, Box<Formula>);

/// Parse a string into a single `Item`
pub fn str_to_item(s: &str) -> Result<Item, ParserError> {
  let parser = grammar::ItemParser::new();
  parser.parse(s).map_err(ParserError::from)
}

/// Parse a string into multiple `Items`
pub fn str_to_items(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::ItemsParser::new();
  parser.parse(s).map_err(ParserError::from)
}

pub fn str_to_relation_type(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::RelationTypeParser::new();
  parser.parse(s).map(|q| q.into()).map_err(ParserError::from)
}

pub fn str_to_rule(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::RuleParser::new();
  parser.parse(s).map(|q| q.into()).map_err(ParserError::from)
}

pub fn str_to_query(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::QueryParser::new();
  parser.parse(s).map(|q| q.into()).map_err(ParserError::from)
}

pub fn str_to_entity(s: &str) -> Result<Entity, ParserError> {
  let parser = grammar::EntityParser::new();
  parser.parse(s).map_err(ParserError::from)
}

pub fn str_to_specialized_predicate(s: &str) -> Result<(Identifier, Vec<Identifier>), ParserError> {
  let parser = grammar::SpecializedPredicateParser::new();
  parser.parse(s).map_err(ParserError::from)
}

pub type RawParserError<'a> = lalrpop_util::ParseError<usize, grammar::Token<'a>, &'static str>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserError {
  pub message: String,
  pub source_id: usize,
  pub source_name: Option<String>,
  pub location_span: Option<(usize, usize)>,
  pub location_point: Option<usize>,
}

impl ParserError {
  pub fn set_source_id(&mut self, id: usize) {
    self.source_id = id;
  }

  pub fn set_source_name(&mut self, name: String) {
    self.source_name = Some(name);
  }
}

impl std::fmt::Display for ParserError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if let Some(source_name) = &self.source_name {
      f.write_fmt(format_args!("Syntax error in {}: {}", source_name, self.message))
    } else {
      f.write_fmt(format_args!("Syntax error: {}", self.message))
    }
  }
}

impl FrontCompileErrorTrait for ParserError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, sources: &Sources) -> String {
    let span = self
      .location_span
      .or_else(|| self.location_point.map(|p| (p, p.saturating_add(1))));
    if let Some((start, end)) = span
      && self.source_id < sources.sources.len()
    {
      let mut location = NodeLocation::from_span(start, end);
      location.source_id = self.source_id;
      format!("{self}\n{}", location.report(sources))
    } else {
      self.to_string()
    }
  }
}

impl<'a> From<RawParserError<'a>> for ParserError {
  fn from(err: RawParserError<'a>) -> Self {
    // Generate base error
    let mut e = ParserError {
      source_id: 0,
      message: format!("{}", err),
      source_name: None,
      location_point: None,
      location_span: None,
    };

    // Add location information
    match err {
      RawParserError::UnrecognizedEof { location, .. } => e.location_point = Some(location),
      RawParserError::InvalidToken { location, .. } => e.location_point = Some(location),
      RawParserError::UnrecognizedToken { token, .. } => e.location_span = Some((token.0, token.2)),
      _ => {}
    }

    // Return the error
    e
  }
}
