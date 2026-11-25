use scallop_core::compiler::front::ast::*;
use scallop_core::compiler::front::parser::*;
use scallop_core::compiler::front::{
  FileSource, FrontContext, LocationSpanAnnotator, ReplSource, Sources, StringSource,
};

#[test]
fn diagnostic_snippets_preserve_source_origins_and_utf8_columns() {
  let text = "skip\n\u{03bb}\u{754c} unexpected";
  let mut sources = Sources::new();
  let file = sources.add(FileSource {
    file_name: "example.scl".into(),
    file_content: text.into(),
    line_offset_len: vec![(0, 4), (5, text.len() - 5)],
  });
  let repl = sources.add(ReplSource::new(42, text.into()));
  let mut location = NodeLocation::from_span("skip\n\u{03bb}".len(), "skip\n\u{03bb}\u{754c}".len());
  location.source_id = file;
  let report = location.report(&sources);
  assert!(report.contains("example.scl:2:2"), "{report}");
  assert!(report.contains("^^"), "{report}");

  location.source_id = repl;
  let report = location.report_warning(&sources);
  assert!(report.contains("REPL:42:2:2"), "{report}");
}

#[test]
fn diagnostic_snippets_can_mark_end_of_input() {
  let report = FrontContext::new()
    .compile_source(ReplSource::new(42, "rel x =".into()))
    .unwrap_err()
    .to_string();
  assert!(
    report.contains("REPL:42:1:8") && report.contains("rel x =") && report.contains('^'),
    "{report}"
  );
}

#[test]
fn source_offset_lookup_preserves_byte_columns_and_duplicate_starts() {
  let source = StringSource::new("\u{03bb}x\nsecond".into());
  let annotator = LocationSpanAnnotator::new(&source);
  assert_eq!(annotator.row_col_of_offset(&2), CharLocation { row: 0, col: 2 });
  assert_eq!(annotator.row_col_of_offset(&4), CharLocation { row: 1, col: 0 });

  let duplicate_offsets = LocationSpanAnnotator {
    row_offset_length: vec![(0, 0), (0, 0), (0, 3), (4, 6)],
  };
  assert_eq!(duplicate_offsets.row_col_of_offset(&0), CharLocation { row: 2, col: 0 });
  let empty = LocationSpanAnnotator {
    row_offset_length: vec![],
  };
  assert_eq!(empty.row_col_of_offset(&0), CharLocation { row: 0, col: 0 });
}

#[test]
fn parse_type_decl() {
  assert!(str_to_item("type symbol = usize").is_ok());
  assert!(str_to_item("type object_id <: symbol").is_ok());
}

#[test]
fn parse_constant_set() {
  assert!(str_to_item("rel edge :- {(0, 1), (1, 2)}").is_ok());
  assert!(str_to_item("rel is_ok :- {(0, true), (1, false)}").is_ok());

  // Probabilities
  assert!(str_to_item(r#"rel edge :- { 0.9::(0, 1), 0.8::(1, 2) }"#).is_ok());
  assert!(str_to_item(r#"rel color :- { 0.9::(0, "blue"), 0.8::(1, "green") }"#).is_ok());
}

#[test]
fn parse_fact() {
  assert!(str_to_item("rel edge(1, 2)").is_ok());
  assert!(str_to_item("rel 0.9::color(0, \"blue\")").is_ok());
}

#[test]
fn parse_rule() {
  assert!(str_to_item(r#"rel path(a, b) :- edge(a, b)"#).is_ok());
  assert!(str_to_item(r#"rel path(a, b) :- path(a, c), edge(c, b)"#).is_ok());
  assert!(str_to_item(r#"rel path(a, b) :- path(a, c) /\ edge(c, b)"#).is_ok());
  assert!(str_to_item(r#"rel path(a, b) :- edge(a, b) \/ path(a, c) /\ edge(c, b)"#).is_ok());
}

#[test]
fn display_range_round_trips() {
  for source in ["rel numbers(x) = x in 0..10", "rel numbers(x) = x in (0 + 1)..=10"] {
    let printed = str_to_item(source).unwrap().to_string();
    assert_eq!(printed, source);
    assert!(str_to_item(&printed).is_ok());
  }
}

#[test]
fn ignore_comment_1() {
  assert!(str_to_item(r#"rel relate = { /* this is a comment */ }"#).is_ok());
  assert!(str_to_item(r#"rel relate = /* this is a comment */ {  }"#).is_ok());
  assert!(str_to_item(r#"rel relate /* this is a comment */ = {  }"#).is_ok());
}

#[test]
fn ignore_comment_2() {
  let items = str_to_items(
    r#"
    rel relate = { /* this is a comment */ }
    rel another_relate() = // this is another comment
      some_atom() /* this is another comment */
  "#,
  )
  .expect("Compile failure");
  assert_eq!(items.len(), 2);
}

#[test]
fn ignore_comment_3() {
  let items = str_to_items(
    r#"
    rel relate = { (3, 5 /* , 4, pretending to be commented out */) }
  "#,
  )
  .expect("Compile failure");
  assert_eq!(items.len(), 1);
}

#[test]
fn test_parse_specialized_predicate_1() {
  let (id, args) = str_to_specialized_predicate("range<usize>").expect("Cannot parse");
  assert_eq!(id.name(), "range");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 1);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 6);
  assert_eq!(args[0].location().offset_span.end, 11);
}

#[test]
fn test_parse_specialized_predicate_2() {
  let (id, args) = str_to_specialized_predicate("range<   usize,usize   >").expect("Cannot parse");
  assert_eq!(id.name(), "range");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 2);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 9);
  assert_eq!(args[0].location().offset_span.end, 14);
  assert_eq!(args[1].name(), "usize");
  assert_eq!(args[1].location().offset_span.start, 15);
  assert_eq!(args[1].location().offset_span.end, 20);
}

#[test]
fn test_parse_specialized_predicate_3() {
  let (id, args) = str_to_specialized_predicate("dasdf<usize, f32>").expect("Cannot parse");
  assert_eq!(id.name(), "dasdf");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 2);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 6);
  assert_eq!(args[0].location().offset_span.end, 11);
  assert_eq!(args[1].name(), "f32");
  assert_eq!(args[1].location().offset_span.start, 13);
  assert_eq!(args[1].location().offset_span.end, 16);
}

#[test]
fn test_parse_specialized_predicate_4() {
  let (id, args) = str_to_specialized_predicate("dasdf  <  usize    , f32 >").expect("Cannot parse");
  assert_eq!(id.name(), "dasdf");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 2);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 10);
  assert_eq!(args[0].location().offset_span.end, 15);
  assert_eq!(args[1].name(), "f32");
  assert_eq!(args[1].location().offset_span.start, 21);
  assert_eq!(args[1].location().offset_span.end, 24);
}
