use clap::Parser;
use std::path::PathBuf;

use scallop_core::compiler;

#[derive(Debug, Default, Parser)]
#[command(name = "sclc", about = "Scallop Compiler", version)]
pub struct Options {
  pub input: PathBuf,

  #[arg(long)]
  pub debug_rs: bool,

  #[arg(long, default_value = "executable")]
  pub mode: String,

  #[arg(long)]
  pub do_not_copy_artifact: bool,

  #[arg(long)]
  pub do_not_keep_temporary_directory: bool,

  #[arg(long)]
  pub dump_rs: bool,

  #[arg(long)]
  pub provenance: Option<String>,

  #[arg(short = 'k', long, default_value_t = 3)]
  pub top_k: usize,

  #[arg(long)]
  pub wmc_with_disjunctions: bool,
}

impl From<&Options> for compiler::CompileOptions {
  fn from(_: &Options) -> Self {
    Self { ..Default::default() }
  }
}
