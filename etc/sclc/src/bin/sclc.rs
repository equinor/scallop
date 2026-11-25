use clap::Parser;
use std::process::ExitCode;

use scallop_core::compiler;
use sclc_core::*;

fn main() -> ExitCode {
  // Command line arguments
  let opt = options::Options::parse();

  // Compile
  let compile_opt = compiler::CompileOptions::from(&opt);
  let ram = match compiler::compile_file_to_ram_with_options(&opt.input, &compile_opt) {
    Ok(ram) => ram,
    Err(errs) => {
      for err in errs {
        eprintln!("{}", err);
      }
      return ExitCode::FAILURE;
    }
  };

  // Depending on the mode, create artifacts
  let result = match opt.mode.as_str() {
    "executable" | "exec" => exec::create_executable(&opt, compile_opt, &ram),
    "pylib" => pylib::create_pylib(&opt, compile_opt, &ram),
    m => {
      eprintln!("Unknown compilation mode --mode `{m}`");
      return ExitCode::FAILURE;
    }
  };
  match result {
    Ok(()) => ExitCode::SUCCESS,
    Err(error) => {
      eprintln!("{error}");
      ExitCode::FAILURE
    }
  }
}
