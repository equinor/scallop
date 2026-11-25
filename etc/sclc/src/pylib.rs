use proc_macro2::TokenStream;
use quote::*;
use scallop_core::common::output_option::OutputOption;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

use scallop_core::compiler;

use super::error::*;
use super::options::*;

pub fn create_pylib(
  opt: &Options,
  compile_opt: compiler::CompileOptions,
  ram: &compiler::ram::Program,
) -> Result<(), SclcError> {
  let (parent_dir, tmp_dir) = generate_pylib_rust_project(&opt.input, compile_opt, ram)?;

  // Compile the file: create command
  let mut cmd = Command::new("maturin");

  // Add arguments
  cmd
    .current_dir(&tmp_dir)
    .arg("build")
    .arg("--release")
    .arg("--out")
    .arg(PathBuf::new().join("target").join("wheels").join("current"));

  // Run the command
  let output = cmd.output()?;
  if output.status.success() {
    let whls =
      fs::read_dir(tmp_dir.join("target").join("wheels").join("current"))?.collect::<std::io::Result<Vec<_>>>()?;
    let [wheel] = whls.as_slice() else {
      return Err(SclcError::Build(
        "maturin",
        format!("expected one wheel, got {}", whls.len()),
      ));
    };
    let whl = wheel.path();
    let whl_name = whl.file_name().unwrap();

    // Copy the wheel to the wheels directory
    fs::copy(&whl, tmp_dir.join("target").join("wheels").join(whl_name))?;

    // If we want to copy the wheel to the parent directory
    if !opt.do_not_copy_artifact {
      fs::copy(&whl, parent_dir.join(whl_name))?;
    }

    // If we want to remove the temporary directory
    if opt.do_not_keep_temporary_directory {
      fs::remove_dir_all(tmp_dir)?;
    }

    Ok(())
  } else {
    Err(SclcError::Build(
      "maturin",
      format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
      ),
    ))
  }
}

pub fn generate_pylib_rust_project(
  input: &Path,
  compile_opt: compiler::CompileOptions,
  ram: &compiler::ram::Program,
) -> Result<(PathBuf, PathBuf), SclcError> {
  // Get program name
  let program_name = input.file_prefix().unwrap().to_str().unwrap();

  // Rust source
  let file_content = generate_pylib_code(compile_opt, program_name, ram)?.to_string();

  // Create a folder
  let parent_dir = input.parent().unwrap();
  let tmp_dir = parent_dir.join(format!(".{}.pylib.sclcmpl", program_name));
  let scallop_source_dir = env::var("SCALLOPDIR").map_err(|_| SclcError::MissingScallopDir)?;

  // Create a temporary directory holding the cargo project
  fs::create_dir_all(tmp_dir.join("src"))?;
  fs::create_dir_all(tmp_dir.join("target").join("wheels").join("current"))?;

  // Create a Cargo.toml file
  let mut cargo_toml_file = File::create(tmp_dir.join("Cargo.toml"))?;
  cargo_toml_file.write_all(
    format!(
      r#"[package]
name = "{}"
version = "1.0.0"
edition = "2024"
[lib]
name = "{}"
crate-type = ["cdylib"]
[dependencies]
scallop-core = {{ path = "{}/core" }}
scallopy-interop = {{ path = "{}/etc/scallopy-interop" }}
rayon = "1.12"
[dependencies.pyo3]
version = "0.30.1"
features = ["extension-module"]
[workspace]
"#,
      program_name, program_name, scallop_source_dir, scallop_source_dir,
    )
    .as_bytes(),
  )?;

  // Create a lib.rs file
  let mut lib_rs_file = File::create(tmp_dir.join("src/lib.rs"))?;
  lib_rs_file.write_all(file_content.as_bytes())?;

  // Return
  Ok((parent_dir.to_path_buf(), tmp_dir))
}

fn generate_pylib_code(
  compile_opt: compiler::CompileOptions,
  program_name: &str,
  ram: &compiler::ram::Program,
) -> Result<TokenStream, SclcError> {
  let program_name_ident = format_ident!("{}", program_name);

  // Turn the ram module into a sequence of rust tokens
  let module = ram.to_rs_module(&compile_opt)?;

  // Generate code
  let create_edb_code = ram.to_rs_create_edb_fn();
  let static_context_code = generate_static_context_code(ram);
  let binding_error_code = generate_binding_error_code();
  let has_file_outputs = ram.relations().any(|r| matches!(r.output, OutputOption::File(_)));
  let context_code = generate_context_code(has_file_outputs);
  let helper_functions = generate_helper_functions();

  // Generate the code
  Ok(quote! {
    use pyo3::{prelude::*, Bound, IntoPyObjectExt};
    use pyo3::exceptions::*;
    use pyo3::types::*;
    use rayon::prelude::*;
    use scallop_core::common::foreign_tensor::{DynamicExternalTensor, FromTensor};
    use scallop_core::common::tuple::*;
    use scallop_core::common::tuple_type::*;
    use scallop_core::common::value::*;
    use scallop_core::common::value_type::*;
    use scallop_core::runtime::provenance::*;
    use scallop_core::runtime::database::extensional::*;
    use scallop_core::utils::*;
    use scallopy_interop::{diff_exclusive_probability_tag, diff_probability_tag, to_py};
    use std::collections::*;
    use std::sync::Arc;
    mod scallop_module { #module #create_edb_code }
    #static_context_code
    #binding_error_code
    #context_code
    #helper_functions
    #[pymodule]
    fn #program_name_ident(m: &Bound<'_, PyModule>) -> PyResult<()> {
      m.add_class::<Context>()?;
      Ok(())
    }
  })
}

fn generate_static_context_code(ram: &compiler::ram::Program) -> TokenStream {
  let get_relation_match_cases = ram
    .relations()
    .filter_map(|r| {
      let relation_name = r.predicate.clone();
      let field_name = r.to_rs_field_name();
      match &r.output {
        OutputOption::Default => Some(quote! {
          #relation_name => Ok(output.#field_name.to_dynamic_vec()),
        }),
        _ => None,
      }
    })
    .collect::<Vec<_>>();

  quote! {
    #[derive(Clone)]
    struct StaticContext<C: PythonProvenance> {
      prov_ctx: C,
      edb: ExtensionalDatabase<C>,
      output: Option<scallop_module::OutputRelations<C>>,
    }

    impl<C: PythonProvenance> StaticContext<C> {
      fn new(prov_ctx: C) -> Self {
        let edb = scallop_module::create_edb();
        Self { prov_ctx, edb, output: None }
      }

      fn add_py_facts(&mut self, r: &str, py_facts: &Bound<'_, PyList>) -> Result<(), BindingError> {
        if let Some(ty) = self.edb.type_of(r) {
          let facts = C::process_typed_py_facts(py_facts, &ty)?;

          // Add facts and disjunctions to relation
          self.edb.add_static_input_facts_without_type_check(r, facts).unwrap();

          // Return Ok
          Ok(())
        } else {
          Err(BindingError(format!("Unknown relation `{}`", r)))
        }
      }

      fn add_facts(&mut self, r: &str, facts: Vec<(Option<C::InputTag>, Tuple)>) -> Result<(), BindingError> {
        if self.edb.type_of(r).is_some() {
          // Add facts and disjunctions to relation
          self.edb.add_static_input_facts(r, facts).unwrap();

          // Return Ok
          Ok(())
        } else {
          Err(BindingError(format!("Unknown relation `{}`", r)))
        }
      }

      fn run(&mut self) -> Result<(), BindingError> {
        self.output = None;
        let edb_clone = self.edb.clone();
        let res = scallop_module::try_run_with_edb(&mut self.prov_ctx, edb_clone, Default::default())
          .map_err(|err| BindingError(err.to_string()))?;
        self.output = Some(res);
        Ok(())
      }

      fn relation(&self, r: &str) -> Result<Vec<(C::OutputTag, Tuple)>, BindingError> {
        if let Some(output) = &self.output {
          match r {
            #(#get_relation_match_cases)*
            r => Err(BindingError(format!("Unknown relation `{}`", r))),
          }
        } else {
          Err(BindingError(format!("There is no execution result yet; please call `.run()` first.")))
        }
      }

      fn py_relation(&self, r: &str) -> Result<Vec<Py<PyAny>>, BindingError> {
        self.relation(r)?
          .into_iter()
          .map(|fact| C::to_output_py_fact(fact).map_err(BindingError::from))
          .collect()
      }
    }
  }
}

fn generate_binding_error_code() -> TokenStream {
  quote! {
    #[derive(Debug)]
    struct BindingError(String);

    impl std::error::Error for BindingError {}

    impl std::fmt::Display for BindingError {
      fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
      }
    }

    impl std::convert::From<BindingError> for PyErr {
      fn from(err: BindingError) -> Self {
        PyException::new_err(err.0)
      }
    }

    impl std::convert::From<PyErr> for BindingError {
      fn from(err: PyErr) -> Self {
        Self(format!("{}", err))
      }
    }
  }
}

fn generate_context_code(has_file_outputs: bool) -> TokenStream {
  quote! {
    #[derive(Clone)]
    enum ContextEnum {
      Unit(StaticContext<unit::UnitProvenance>),
      MinMaxProb(StaticContext<min_max_prob::MinMaxProbProvenance>),
      AddMultProb(StaticContext<add_mult_prob::AddMultProbProvenance>),
      DiffMinMaxProb(StaticContext<diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, ArcFamily>>),
      DiffTopKProofs(StaticContext<diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, ArcFamily>>),
      DiffTopBottomKClauses(StaticContext<diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, ArcFamily>>),
    }

    #[pyclass(unsendable, name = "StaticContext")]
    struct Context { ctx: ContextEnum }

    #[pymethods]
    impl Context {
      #[new]
      #[pyo3(signature = (provenance = "unit", top_k = None, wmc_with_disjunctions = false))]
      fn new(provenance: &str, top_k: Option<usize>, wmc_with_disjunctions: bool) -> PyResult<Self> {
        let top_k = top_k.unwrap_or(3);
        match provenance {
          "unit" => Ok(Self { ctx: ContextEnum::Unit(StaticContext::new(unit::UnitProvenance::default())) }),
          "minmaxprob" => Ok(Self { ctx: ContextEnum::MinMaxProb(StaticContext::new(min_max_prob::MinMaxProbProvenance::default())) }),
          "addmultprob" => Ok(Self { ctx: ContextEnum::AddMultProb(StaticContext::new(add_mult_prob::AddMultProbProvenance::default())) }),
          "diffminmaxprob" => Ok(Self { ctx: ContextEnum::DiffMinMaxProb(StaticContext::new(diff_min_max_prob::DiffMinMaxProbProvenance::default())) }),
          "difftopkproofs" => Ok(Self { ctx: ContextEnum::DiffTopKProofs(StaticContext::new(diff_top_k_proofs::DiffTopKProofsProvenance::new(top_k, wmc_with_disjunctions))) }),
          "difftopbottomkclauses" => Ok(Self { ctx: ContextEnum::DiffTopBottomKClauses(StaticContext::new(diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance::new(top_k, wmc_with_disjunctions))) }),
          p => Err(PyErr::from(BindingError(format!("Unknown provenance `{}`", p.to_string())))),
        }
      }

      fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        match &self.ctx {
          ContextEnum::Unit(_) => None,
          ContextEnum::MinMaxProb(_) => None,
          ContextEnum::AddMultProb(_) => None,
          ContextEnum::DiffMinMaxProb(_) => None,
          ContextEnum::DiffTopKProofs(c) => Some(c.prov_ctx.input_tags().into_vec()),
          ContextEnum::DiffTopBottomKClauses(c) => Some(c.prov_ctx.input_tags().into_vec()),
        }
      }

      fn add_facts(&mut self, r: &str, facts: &Bound<'_, PyList>) -> PyResult<()> {
        let res = match &mut self.ctx {
          ContextEnum::Unit(c) => c.add_py_facts(r, facts),
          ContextEnum::MinMaxProb(c) => c.add_py_facts(r, facts),
          ContextEnum::AddMultProb(c) => c.add_py_facts(r, facts),
          ContextEnum::DiffMinMaxProb(c) => c.add_py_facts(r, facts),
          ContextEnum::DiffTopKProofs(c) => c.add_py_facts(r, facts),
          ContextEnum::DiffTopBottomKClauses(c) => c.add_py_facts(r, facts),
        };
        res.map_err(PyErr::from)
      }

      fn run(&mut self) -> PyResult<()> {
        match &mut self.ctx {
          ContextEnum::Unit(c) => c.run(),
          ContextEnum::MinMaxProb(c) => c.run(),
          ContextEnum::AddMultProb(c) => c.run(),
          ContextEnum::DiffMinMaxProb(c) => c.run(),
          ContextEnum::DiffTopKProofs(c) => c.run(),
          ContextEnum::DiffTopBottomKClauses(c) => c.run(),
        }.map_err(PyErr::from)
      }

      fn run_batch_parallel(&self, py: Python<'_>, inputs: HashMap<String, Vec<Bound<'_, PyList>>>, outputs: Vec<String>) -> PyResult<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>> {
        if #has_file_outputs {
          // Batch items share the same configured output paths.
          return self.run_batch_non_parallel(inputs, outputs);
        }
        let res = match &self.ctx {
          ContextEnum::Unit(c) => run_batch_parallel(py, c, inputs, outputs),
          ContextEnum::MinMaxProb(c) => run_batch_parallel(py, c, inputs, outputs),
          ContextEnum::AddMultProb(c) => run_batch_parallel(py, c, inputs, outputs),
          ContextEnum::DiffMinMaxProb(c) => run_batch_parallel(py, c, inputs, outputs),
          ContextEnum::DiffTopKProofs(c) => run_batch_parallel(py, c, inputs, outputs),
          ContextEnum::DiffTopBottomKClauses(c) => run_batch_parallel(py, c, inputs, outputs),
        };
        res.map_err(PyErr::from)
      }

      fn run_batch_non_parallel(&self, inputs: HashMap<String, Vec<Bound<'_, PyList>>>, outputs: Vec<String>) -> PyResult<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>> {
        let res = match &self.ctx {
          ContextEnum::Unit(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::MinMaxProb(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::AddMultProb(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::DiffMinMaxProb(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::DiffTopKProofs(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::DiffTopBottomKClauses(c) => run_batch_non_parallel(c, inputs, outputs),
        };
        res.map_err(PyErr::from)
      }

      fn relation(&self, r: &str) -> PyResult<Vec<Py<PyAny>>> {
        let res = match &self.ctx {
          ContextEnum::Unit(c) => c.py_relation(r),
          ContextEnum::MinMaxProb(c) => c.py_relation(r),
          ContextEnum::AddMultProb(c) => c.py_relation(r),
          ContextEnum::DiffMinMaxProb(c) => c.py_relation(r),
          ContextEnum::DiffTopKProofs(c) => c.py_relation(r),
          ContextEnum::DiffTopBottomKClauses(c) => c.py_relation(r),
        };
        res.map_err(PyErr::from)
      }
    }
  }
}

fn generate_helper_functions() -> TokenStream {
  quote! {
    fn run_batch_parallel<C>(
      py: Python<'_>,
      c: &StaticContext<C>,
      inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
      outputs: Vec<String>,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError>
    where
      C: PythonProvenance + std::marker::Sync,
      <C as Provenance>::InputTag: std::marker::Sync,
      <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
      <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();

      // Create contexts for parallel computing
      let prep_inputs = inputs
        .into_iter()
        .map(|(r, batch)| {
          let ty = c.edb.type_of(&r).ok_or(BindingError(format!("Unknown relation `{}`", r)))?;
          let prep_batch = batch.into_iter().map(|facts| C::process_typed_py_facts(&facts, &ty)).collect::<Result<_, BindingError>>()?;
          Ok((r, prep_batch))
        })
        .collect::<Result<HashMap<String, Vec<Vec<(Option<C::InputTag>, Tuple)>>>, BindingError>>()?;

      // Actually execute the program in parallel
      let base_results = py.detach(|| (0..batch_size)
        .into_par_iter()
        .map(|i| {
          let mut temp_ctx = c.clone();
          for (r, batch) in prep_inputs.iter() {
            temp_ctx.add_facts(r, batch[i].clone())?;
          }
          temp_ctx.run()?;
          let input_tags = C::get_input_tags(&temp_ctx.prov_ctx);
          let relations = outputs.iter().map(|o| temp_ctx.relation(o)).collect::<Result<Vec<_>, _>>()?;
          Ok((input_tags, relations))
        })
        .collect::<Result<Vec<_>, BindingError>>())?;

      to_python_batch::<C>(base_results)
    }

    fn run_batch_non_parallel<C>(
      c: &StaticContext<C>,
      inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
      outputs: Vec<String>,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError>
    where
      C: PythonProvenance,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();

      // Create contexts for parallel computing
      let prep_inputs = inputs
        .into_iter()
        .map(|(r, batch)| {
          let ty = c.edb.type_of(&r).ok_or(BindingError(format!("Unknown relation `{}`", r)))?;
          let prep_batch = batch.into_iter().map(|facts| C::process_typed_py_facts(&facts, &ty)).collect::<Result<_, BindingError>>()?;
          Ok((r, prep_batch))
        })
        .collect::<Result<HashMap<String, Vec<Vec<(Option<C::InputTag>, Tuple)>>>, BindingError>>()?;

      // Actually execute the program in parallel
      let base_results = (0..batch_size)
        .into_iter()
        .map(|i| {
          let mut temp_ctx = c.clone();
          for (r, batch) in prep_inputs.iter() {
            temp_ctx.add_facts(r, batch[i].clone())?;
          }
          temp_ctx.run()?;
          let input_tags = C::get_input_tags(&temp_ctx.prov_ctx);
          let relations = outputs.iter().map(|o| temp_ctx.relation(o)).collect::<Result<Vec<_>, _>>()?;
          Ok((input_tags, relations))
        })
        .collect::<Result<Vec<_>, BindingError>>()?;

      to_python_batch::<C>(base_results)
    }

    fn to_python_batch<C: PythonProvenance>(
      batch: Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<(C::OutputTag, Tuple)>>)>,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError> {
      batch
        .into_iter()
        .map(|(input_tags, relations)| {
          let relations = relations
            .into_iter()
            .map(|rs| rs.into_iter().map(C::to_output_py_fact).collect())
            .collect::<PyResult<_>>()?;
          Ok((input_tags, relations))
        })
        .collect()
    }

    fn to_python_tuple(tup: &Tuple) -> PyResult<Py<PyAny>> {
      Python::attach(|py| scallopy_interop::to_python_tuple(py, tup, &|py, value| {
        match value {
          Value::SymbolString(symbol) => symbol.into_py_any(py),
          Value::DateTime(time) => time.to_string().into_py_any(py),
          Value::Duration(duration) => duration.to_string().into_py_any(py),
          Value::Entity(id) => id.into_py_any(py),
          _ => Err(PyTypeError::new_err(format!("Unsupported compiled output value `{value}`"))),
        }
      }))
    }

    fn from_python_tuple(v: &Bound<'_, PyAny>, ty: &TupleType) -> PyResult<Tuple> {
      scallopy_interop::from_python_tuple(v, ty, &|v, ty| {
        match ty {
          ValueType::Symbol => v.extract().map(Value::SymbolString),
          ValueType::DateTime => {
            let text: &str = v.extract()?;
            parse_date_time_string(text).map(Value::DateTime)
              .ok_or_else(|| PyTypeError::new_err(format!("Cannot parse into DateTime: {text}")))
          }
          ValueType::Duration => {
            let text: &str = v.extract()?;
            parse_duration_string(text).map(Value::Duration)
              .ok_or_else(|| PyTypeError::new_err(format!("Cannot parse into Duration: {text}")))
          }
          ValueType::Entity if v.cast::<PyString>().is_ok() => v.extract().map(Value::EntityString),
          ValueType::Entity => v.extract().map(Value::Entity),
          _ => Err(PyTypeError::new_err(format!("Unsupported compiled input type `{ty}`"))),
        }
      })
    }

    #[derive(Clone)]
    pub struct ExtTag {
      tag: Arc<Py<PyAny>>,
    }

    impl FromTensor for ExtTag {
      fn from_tensor(_: DynamicExternalTensor) -> Option<Self> {
        None
      }
    }

    impl From<&Bound<'_, PyAny>> for ExtTag {
      fn from(tag: &Bound<'_, PyAny>) -> Self {
        Self { tag: Arc::new(tag.clone().unbind()) }
      }
    }

    impl From<ExtTag> for Py<PyAny> {
      fn from(tag: ExtTag) -> Self {
        Arc::try_unwrap(tag.tag).unwrap_or_else(|tag| Python::attach(|py| tag.clone_ref(py)))
      }
    }

    pub trait ExtTagVec {
      fn into_vec(self) -> Vec<Py<PyAny>>;
    }

    impl ExtTagVec for Vec<ExtTag> {
      fn into_vec(self) -> Vec<Py<PyAny>> {
        self.into_iter().map(Into::into).collect()
      }
    }

    trait PythonProvenance: Provenance {
      /// Process a list of python facts while the tuples are typed with `tuple_type`
      fn process_typed_py_facts(facts: &Bound<'_, PyList>, tuple_type: &TupleType) -> Result<Vec<(Option<Self::InputTag>, Tuple)>, BindingError> {
        facts
          .iter()
          .map(|fact| {
            let (maybe_py_tag, py_tup) = Self::split_py_fact(fact)?;
            let tag = Self::process_optional_py_tag(maybe_py_tag.as_ref())?;
            let tup = from_python_tuple(&py_tup, tuple_type).map_err(|err| BindingError(format!("Cannot parse tuple `{}` with type `{}`: {}", py_tup, tuple_type, err)))?;
            Ok((tag, tup))
          })
          .collect::<Result<Vec<_>, _>>()
      }

      /// Split a python object into (Option<PyTag>, PyTup) pair
      fn split_py_fact(fact: Bound<'_, PyAny>) -> Result<(Option<Bound<'_, PyAny>>, Bound<'_, PyAny>), BindingError> {
        let (py_tag, py_tup) = fact.extract()?;
        Ok((py_tag, py_tup))
      }

      /// Convert a python object into an optional input tag for this provenance context
      fn process_optional_py_tag(maybe_py_tag: Option<&Bound<'_, PyAny>>) -> Result<Option<Self::InputTag>, BindingError> {
        if let Some(py_tag) = maybe_py_tag {
          Ok(Self::process_py_tag(py_tag)?)
        } else {
          Ok(None)
        }
      }

      /// Convert a python object into an optional input tag for this provenance context
      fn process_py_tag(tag: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError>;

      /// Convert an output tag into a python object
      fn to_output_py_tag(tag: &Self::OutputTag) -> PyResult<Py<PyAny>>;

      /// Convert a fact into a python fact
      fn to_output_py_fact((tag, tup): (Self::OutputTag, Tuple)) -> PyResult<Py<PyAny>> {
        to_py((Self::to_output_py_tag(&tag)?, to_python_tuple(&tup)?))
      }

      /// Get input tags
      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        None
      }
    }

    impl PythonProvenance for unit::UnitProvenance {
      fn split_py_fact(fact: Bound<'_, PyAny>) -> Result<(Option<Bound<'_, PyAny>>, Bound<'_, PyAny>), BindingError> {
        Ok((None, fact))
      }

      fn process_py_tag(_: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        Ok(None)
      }

      fn to_output_py_tag(_: &Self::OutputTag) -> PyResult<Py<PyAny>> {
        to_py(())
      }

      fn to_output_py_fact((_, tup): (Self::OutputTag, Tuple)) -> PyResult<Py<PyAny>> {
        to_python_tuple(&tup)
      }
    }

    impl PythonProvenance for min_max_prob::MinMaxProbProvenance {
      fn process_py_tag(tag: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        tag.extract().map(Some).map_err(BindingError::from)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> PyResult<Py<PyAny>> {
        to_py(tag)
      }
    }

    impl PythonProvenance for add_mult_prob::AddMultProbProvenance {
      fn process_py_tag(tag: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        tag.extract().map(Some).map_err(BindingError::from)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> PyResult<Py<PyAny>> {
        to_py(tag)
      }
    }

    impl PythonProvenance for diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        Ok(diff_probability_tag(tag, |tag| tag.extract())?)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> PyResult<Py<PyAny>> {
        match tag.2 {
          0 => to_py((0, tag.0)),
          sign => {
            let value = tag.3.clone().ok_or_else(|| PyValueError::new_err("Missing differentiable output tag"))?;
            to_py((if sign == 1 { 1 } else { -1 }, Py::from(value)))
          }
        }
      }
    }

    impl PythonProvenance for diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        Ok(diff_exclusive_probability_tag(tag, |tag| tag.extract())?)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> PyResult<Py<PyAny>> {
        to_py((tag.0, tag.1.clone()))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &Bound<'_, PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        Ok(diff_exclusive_probability_tag(tag, |tag| tag.extract())?)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> PyResult<Py<PyAny>> {
        to_py((tag.0, tag.1.clone()))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }
  }
}
