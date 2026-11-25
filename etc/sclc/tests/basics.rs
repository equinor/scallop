mod common;

#[test]
#[ignore]
fn pylib_edge_path() {
  common::check_compile_pylib_from_program_string(
    "edge_path_pylib_1",
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
      type inputs(Symbol, DateTime, Duration)
      rel inputs = {(s"start", t"2020-01-31T12:30:00.125Z", d"1mo -1d")}
      rel shifted(s, t + d) = inputs(s, t, d)
    "#,
  )
}

#[test]
#[ignore]
fn exec_edge_path() {
  common::check_compile_exec_from_program_string(
    "edge_path_exec_1",
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
    "#,
  )
}

#[test]
#[ignore]
fn exec_csv_round_trip_and_missing_input_error() {
  use std::{fs, process::Command};

  let temp = tempfile::tempdir().unwrap();
  let input_csv = temp.path().join("input.csv");
  let output_csv = temp.path().join("output.csv");
  let input = temp.path().join("csv_round_trip.scl");
  fs::write(&input_csv, "key;value\n2;7\n4;9\n").unwrap();
  fs::write(
    &input,
    format!(
      r#"
    @file("{}", deliminator=";", header=true)
    type inputs(i32, i32)
    rel result(key, value + 1) = inputs(key, value)
    @file("{}", deliminator=";")
    query result
  "#,
      input_csv.display(),
      output_csv.display()
    ),
  )
  .unwrap();

  let opts = sclc_core::options::Options {
    input,
    provenance: Some("unit".into()),
    ..Default::default()
  };
  let compiler_opts = scallop_core::compiler::CompileOptions::default();
  let ram = scallop_core::compiler::compile_file_to_ram(&opts.input).unwrap();
  let (_, project) = sclc_core::exec::generate_exec_rust_project(&opts, compiler_opts, &ram).unwrap();
  let run = || {
    Command::new("cargo")
      .current_dir(&project)
      .args(["run", "--quiet"])
      .output()
      .unwrap()
  };

  let output = run();
  assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
  assert_eq!(fs::read_to_string(output_csv).unwrap(), "2;8\n4;10\n");
  fs::remove_file(input_csv).unwrap();
  let output = run();
  assert!(!output.status.success());
  assert!(String::from_utf8_lossy(&output.stderr).contains("input.csv"));
}
