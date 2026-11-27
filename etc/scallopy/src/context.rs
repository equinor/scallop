use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;

use rayon::prelude::*;

use scallop_core::common::tuple::*;
use scallop_core::common::tuple_type::*;
use scallop_core::integrate::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::monitor;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::*;
use scallopy_interop::from_optional_python_tag;

use crate::custom_tag;

use super::collection::*;
use super::error::*;
use super::external_tag::*;
use super::foreign_attribute::*;
use super::foreign_function::*;
use super::foreign_predicate::*;
use super::io::*;
use super::provenance::*;
use super::tensor::*;
use super::tuple::*;

type AF = ArcFamily;
type WeightedContext = IntegrateContext<top_k_proofs::TopKProofsProvenance<AF>, AF>;
type DiffWeightedContext = IntegrateContext<diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, AF>, AF>;
type PyBoundsInterval = (f64, f64, usize, &'static str);
type PyGradientBound = (Py<PyAny>, f64, f64, Option<f64>);
type PyBoundsResult = (
  PyBoundsInterval,
  Option<f64>,
  Vec<PyGradientBound>,
  Option<Vec<(usize, usize, f64)>>,
);
type BoundsFact = (Option<InputExclusiveDiffProb<ExtTag>>, Tuple);
type PyJointOutcome<'py> = (Bound<'py, PyAny>, BTreeMap<String, Vec<Bound<'py, PyAny>>>);

#[derive(Clone)]
pub struct JointOutcome {
  weight: InputExclusiveDiffProb<ExtTag>,
  rows: Vec<(String, Tuple)>,
}

#[derive(Clone)]
pub enum ContextEnum {
  // Basic
  Unit(IntegrateContext<unit::UnitProvenance, AF>),
  Proofs(IntegrateContext<proofs::ProofsProvenance<AF>, AF>),
  Tropical(IntegrateContext<tropical::TropicalProvenance, AF>),

  // Probabilistic
  MinMaxProb(IntegrateContext<min_max_prob::MinMaxProbProvenance, AF>),
  AddMultProb(IntegrateContext<add_mult_prob::AddMultProbProvenance, AF>),
  TopKProofs(WeightedContext),
  TopBottomKClauses(IntegrateContext<top_bottom_k_clauses::TopBottomKClausesProvenance<AF>, AF>),
  // Reuse weighted input parsing; bounded evaluation never executes this context.
  AnytimeBounds(DiffWeightedContext, Vec<Vec<JointOutcome>>),

  // Differentiable
  DiffMinMaxProb(IntegrateContext<diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, AF>, AF>),
  DiffAddMultProb(IntegrateContext<diff_add_mult_prob::DiffAddMultProbProvenance<ExtTag, AF>, AF>),
  DiffNandMultProb(IntegrateContext<diff_nand_mult_prob::DiffNandMultProbProvenance<ExtTag, AF>, AF>),
  DiffMaxMultProb(IntegrateContext<diff_max_mult_prob::DiffMaxMultProbProvenance<ExtTag, AF>, AF>),
  DiffNandMinProb(IntegrateContext<diff_nand_min_prob::DiffNandMinProbProvenance<ExtTag, AF>, AF>),
  DiffSampleKProofs(IntegrateContext<diff_sample_k_proofs::DiffSampleKProofsProvenance<ExtTag, AF>, AF>),
  DiffTopKProofs(DiffWeightedContext),
  DiffTopBottomKClauses(IntegrateContext<diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, AF>, AF>),

  // Debug
  DiffTopKProofsDebug(IntegrateContext<diff_top_k_proofs_debug::DiffTopKProofsDebugProvenance<ExtTag, AF>, AF>),

  // Custom
  Custom(IntegrateContext<custom_tag::CustomProvenance, AF>),
}

macro_rules! match_context {
  ($ctx:expr, $v:ident, $e:expr) => {
    match $ctx {
      ContextEnum::Unit($v) => $e,
      ContextEnum::Proofs($v) => $e,
      ContextEnum::Tropical($v) => $e,
      ContextEnum::MinMaxProb($v) => $e,
      ContextEnum::AddMultProb($v) => $e,
      ContextEnum::TopKProofs($v) => $e,
      ContextEnum::TopBottomKClauses($v) => $e,
      ContextEnum::AnytimeBounds($v, _) => $e,
      ContextEnum::DiffMinMaxProb($v) => $e,
      ContextEnum::DiffAddMultProb($v) => $e,
      ContextEnum::DiffNandMultProb($v) => $e,
      ContextEnum::DiffMaxMultProb($v) => $e,
      ContextEnum::DiffNandMinProb($v) => $e,
      ContextEnum::DiffSampleKProofs($v) => $e,
      ContextEnum::DiffTopKProofs($v) => $e,
      ContextEnum::DiffTopBottomKClauses($v) => $e,
      ContextEnum::DiffTopKProofsDebug($v) => $e,
      ContextEnum::Custom($v) => $e,
    }
  };
}

macro_rules! match_context_except_custom {
  ($ctx:expr, $v:ident, $e:expr) => {
    match $ctx {
      ContextEnum::Unit($v) => Ok($e),
      ContextEnum::Proofs($v) => Ok($e),
      ContextEnum::Tropical($v) => Ok($e),
      ContextEnum::MinMaxProb($v) => Ok($e),
      ContextEnum::AddMultProb($v) => Ok($e),
      ContextEnum::TopKProofs($v) => Ok($e),
      ContextEnum::TopBottomKClauses($v) => Ok($e),
      ContextEnum::DiffMinMaxProb($v) => Ok($e),
      ContextEnum::DiffAddMultProb($v) => Ok($e),
      ContextEnum::DiffNandMultProb($v) => Ok($e),
      ContextEnum::DiffMaxMultProb($v) => Ok($e),
      ContextEnum::DiffNandMinProb($v) => Ok($e),
      ContextEnum::DiffSampleKProofs($v) => Ok($e),
      ContextEnum::DiffTopKProofs($v) => Ok($e),
      ContextEnum::DiffTopBottomKClauses($v) => Ok($e),
      ContextEnum::DiffTopKProofsDebug($v) => Ok($e),
      ContextEnum::Custom(_) => Err(BindingError::CustomProvenanceUnsupported),
      ContextEnum::AnytimeBounds(..) => Err(bounds_operation_error()),
    }
  };
}

#[pyclass(unsendable, name = "InternalScallopContext")]
pub struct Context {
  pub ctx: ContextEnum,
}

#[pyclass(name = "InternalBoundsSession")]
pub struct BoundsSession(anytime_bounds::BoundsSession);

#[pymethods]
impl BoundsSession {
  #[pyo3(signature=(max_facts, timeout=None))]
  fn advance(&mut self, py: Python<'_>, max_facts: usize, timeout: Option<f64>) -> PyResult<PyBoundsInterval> {
    let timeout = timeout
      .map(Duration::try_from_secs_f64)
      .transpose()
      .map_err(|_| PyValueError::new_err("timeout must be a finite, non-negative duration in seconds"))?;
    let result = py
      .detach(|| {
        if let Some(timeout) = timeout {
          let start = Instant::now();
          self.0.advance_until(max_facts, || start.elapsed() >= timeout)
        } else {
          self.0.advance(max_facts)
        }
      })
      .map_err(py_bounds_error)?;
    Ok(py_bounds_interval(&result))
  }
}

#[pymethods]
impl Context {
  /// Create a new scallop context
  ///
  /// # Arguments
  ///
  /// * `provenance` - a string of the name of the provenance to use for this context
  /// * `k` - an unsigned integer serving as the hyper-parameter for provenance such as `"topkproofs"`
  /// * `custom_provenance` - an optional python object serving as the provenance context
  #[new]
  #[pyo3(signature=(provenance="unit", k=3, *, custom_provenance=None))]
  fn new(provenance: &str, k: usize, custom_provenance: Option<Py<PyAny>>) -> Result<Self, BindingError> {
    // Check provenance type
    match provenance {
      "unit" => Ok(Self {
        ctx: ContextEnum::Unit(IntegrateContext::new_incremental(unit::UnitProvenance)),
      }),
      "proofs" => Ok(Self {
        ctx: ContextEnum::Proofs(IntegrateContext::new_incremental(proofs::ProofsProvenance::default())),
      }),
      "tropical" => Ok(Self {
        ctx: ContextEnum::Tropical(IntegrateContext::new_incremental(tropical::TropicalProvenance)),
      }),
      "minmaxprob" => Ok(Self {
        ctx: ContextEnum::MinMaxProb(IntegrateContext::new_incremental(
          min_max_prob::MinMaxProbProvenance::default(),
        )),
      }),
      "addmultprob" => Ok(Self {
        ctx: ContextEnum::AddMultProb(IntegrateContext::new_incremental(
          add_mult_prob::AddMultProbProvenance::default(),
        )),
      }),
      "topkproofs" => Ok(Self {
        ctx: ContextEnum::TopKProofs(IntegrateContext::new_incremental(
          top_k_proofs::TopKProofsProvenance::new(k),
        )),
      }),
      "topbottomkclauses" => Ok(Self {
        ctx: ContextEnum::TopBottomKClauses(IntegrateContext::new_incremental(
          top_bottom_k_clauses::TopBottomKClausesProvenance::new(k),
        )),
      }),
      "anytimebounds" => Ok(Self {
        ctx: ContextEnum::AnytimeBounds(
          IntegrateContext::new_incremental(diff_top_k_proofs::DiffTopKProofsProvenance::new(0)),
          Vec::new(),
        ),
      }),
      "diffminmaxprob" => Ok(Self {
        ctx: ContextEnum::DiffMinMaxProb(IntegrateContext::new_incremental(
          diff_min_max_prob::DiffMinMaxProbProvenance::default(),
        )),
      }),
      "diffaddmultprob" => Ok(Self {
        ctx: ContextEnum::DiffAddMultProb(IntegrateContext::new_incremental(
          diff_add_mult_prob::DiffAddMultProbProvenance::default(),
        )),
      }),
      "diffnandmultprob" => Ok(Self {
        ctx: ContextEnum::DiffNandMultProb(IntegrateContext::new_incremental(
          diff_nand_mult_prob::DiffNandMultProbProvenance::default(),
        )),
      }),
      "diffmaxmultprob" => Ok(Self {
        ctx: ContextEnum::DiffMaxMultProb(IntegrateContext::new_incremental(
          diff_max_mult_prob::DiffMaxMultProbProvenance::default(),
        )),
      }),
      "diffnandminprob" => Ok(Self {
        ctx: ContextEnum::DiffNandMinProb(IntegrateContext::new_incremental(
          diff_nand_min_prob::DiffNandMinProbProvenance::default(),
        )),
      }),
      "diffsamplekproofs" => Ok(Self {
        ctx: ContextEnum::DiffSampleKProofs(IntegrateContext::new_incremental(
          diff_sample_k_proofs::DiffSampleKProofsProvenance::new(k),
        )),
      }),
      "difftopkproofs" => Ok(Self {
        ctx: ContextEnum::DiffTopKProofs(IntegrateContext::new_incremental(
          diff_top_k_proofs::DiffTopKProofsProvenance::new(k),
        )),
      }),
      "difftopbottomkclauses" => Ok(Self {
        ctx: ContextEnum::DiffTopBottomKClauses(IntegrateContext::new_incremental(
          diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance::new(k),
        )),
      }),
      "difftopkproofsdebug" => Ok(Self {
        ctx: ContextEnum::DiffTopKProofsDebug(IntegrateContext::new_incremental(
          diff_top_k_proofs_debug::DiffTopKProofsDebugProvenance::new(k),
        )),
      }),
      "custom" => {
        if let Some(py_ctx) = custom_provenance {
          Ok(Self {
            ctx: ContextEnum::Custom(IntegrateContext::new_incremental(custom_tag::CustomProvenance(
              py_ctx.into(),
            ))),
          })
        } else {
          Err(BindingError::InvalidCustomProvenance)
        }
      }
      p => Err(BindingError::UnknownProvenance(p.to_string())),
    }
  }

  /// Creates a clone of the scallopy context
  fn clone(&self) -> Self {
    Self { ctx: self.ctx.clone() }
  }

  /// Create a new scallop context with a different provenance as the current context
  fn clone_with_new_provenance(&self, provenance: &str, k: usize) -> Result<Self, BindingError> {
    // Check provenance type
    match provenance {
      "unit" => Ok(Self {
        ctx: ContextEnum::Unit(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(unit::UnitProvenance)
        )?),
      }),
      "proofs" => Ok(Self {
        ctx: ContextEnum::Proofs(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(proofs::ProofsProvenance::default())
        )?),
      }),
      "minmaxprob" => Ok(Self {
        ctx: ContextEnum::MinMaxProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(min_max_prob::MinMaxProbProvenance::default())
        )?),
      }),
      "addmultprob" => Ok(Self {
        ctx: ContextEnum::AddMultProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(add_mult_prob::AddMultProbProvenance::default())
        )?),
      }),
      "topkproofs" => Ok(Self {
        ctx: ContextEnum::TopKProofs(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(top_k_proofs::TopKProofsProvenance::new(k))
        )?),
      }),
      "topbottomkclauses" => Ok(Self {
        ctx: ContextEnum::TopBottomKClauses(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(top_bottom_k_clauses::TopBottomKClausesProvenance::new(k))
        )?),
      }),
      "diffminmaxprob" => Ok(Self {
        ctx: ContextEnum::DiffMinMaxProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_min_max_prob::DiffMinMaxProbProvenance::default(),)
        )?),
      }),
      "diffaddmultprob" => Ok(Self {
        ctx: ContextEnum::DiffAddMultProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_add_mult_prob::DiffAddMultProbProvenance::default(),)
        )?),
      }),
      "diffnandmultprob" => Ok(Self {
        ctx: ContextEnum::DiffNandMultProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_nand_mult_prob::DiffNandMultProbProvenance::default(),)
        )?),
      }),
      "diffmaxmultprob" => Ok(Self {
        ctx: ContextEnum::DiffMaxMultProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_max_mult_prob::DiffMaxMultProbProvenance::default(),)
        )?),
      }),
      "diffnandminprob" => Ok(Self {
        ctx: ContextEnum::DiffNandMinProb(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_nand_min_prob::DiffNandMinProbProvenance::default(),)
        )?),
      }),
      "diffsamplekproofs" => Ok(Self {
        ctx: ContextEnum::DiffSampleKProofs(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_sample_k_proofs::DiffSampleKProofsProvenance::new(k),)
        )?),
      }),
      "difftopkproofs" => Ok(Self {
        ctx: ContextEnum::DiffTopKProofs(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_top_k_proofs::DiffTopKProofsProvenance::new(k))
        )?),
      }),
      "difftopbottomkclauses" => Ok(Self {
        ctx: ContextEnum::DiffTopBottomKClauses(match_context_except_custom!(
          &self.ctx,
          c,
          c.clone_with_new_provenance(diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance::new(k))
        )?),
      }),
      "custom" => Err(BindingError::CustomProvenanceUnsupported),
      p => Err(BindingError::UnknownProvenance(p.to_string())),
    }
  }

  fn enable_tensor_registry(&mut self) {
    match_context!(&mut self.ctx, c, c.set_tensor_registry(TensorRegistry::new()))
  }

  fn set_debug_front(&mut self, debug_front: bool) {
    match_context!(&mut self.ctx, c, c.set_debug_front(debug_front))
  }

  fn set_debug_back(&mut self, debug_back: bool) {
    match_context!(&mut self.ctx, c, c.set_debug_back(debug_back))
  }

  fn set_debug_ram(&mut self, debug_ram: bool) {
    match_context!(&mut self.ctx, c, c.set_debug_ram(debug_ram))
  }

  /// Set the current context to be non-incremental
  fn set_non_incremental(&mut self) {
    match_context!(&mut self.ctx, c, c.set_non_incremental())
  }

  /// Set early discard
  fn set_early_discard(&mut self, early_discard: bool) {
    match_context!(&mut self.ctx, c, c.set_early_discard(early_discard))
  }

  /// Set the iteration limit to be `k`
  fn set_iter_limit(&mut self, k: usize) {
    match_context!(&mut self.ctx, c, c.set_iter_limit(k))
  }

  /// Remove the iteration limit
  fn remove_iter_limit(&mut self) {
    match_context!(&mut self.ctx, c, c.remove_iter_limit())
  }

  /// Add monitors to the system
  fn add_monitors(&mut self, monitors: Vec<String>) {
    let monitors = monitors.iter().map(String::as_str).collect::<Vec<_>>();
    match_context!(&mut self.ctx, c, c.add_monitors(&monitors))
  }

  /// Load monitors to the system (discarding all existing ones)
  fn load_monitors(&mut self, monitors: Vec<String>) {
    let monitors = monitors.iter().map(String::as_str).collect::<Vec<_>>();
    match_context!(&mut self.ctx, c, c.load_monitors(&monitors))
  }

  /// Compile the surface program stored in the scallopy context into the ram program.
  ///
  /// This function is usually used before creating a forward function.
  /// If called multiple times, the ram program compiled later will overwrite the previous ones.
  fn compile(&mut self) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.compile().map_err(BindingError::from))
  }

  /// Import a module written in scallop file
  fn import_file(&mut self, file_name: &str) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.import_file(file_name).map_err(BindingError::from))
  }

  /// Add a program in the form of string
  fn add_program(&mut self, program: &str) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.add_program(program).map_err(BindingError::from))
  }

  /// Add an item and check whether it is a query
  fn add_item(&mut self, item: &str) -> Result<Vec<String>, BindingError> {
    match_context!(&mut self.ctx, c, c.add_item(item).map_err(BindingError::from))
  }

  /// Print the scallop program
  fn dump_front_ir(&self) {
    match_context!(&self.ctx, c, c.dump_front_ir())
  }

  /// Return the scallop program in its string form
  fn get_front_ir(&self) -> String {
    match_context!(&self.ctx, c, c.get_front_ir())
  }

  /// Obtain the list of input tags, if the underlying provenance supports it.
  fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
    match &self.ctx {
      ContextEnum::Unit(_) => None,
      ContextEnum::Proofs(_) => None,
      ContextEnum::Tropical(_) => None,
      ContextEnum::MinMaxProb(_) => None,
      ContextEnum::AddMultProb(_) => None,
      ContextEnum::TopKProofs(_) => None,
      ContextEnum::TopBottomKClauses(_) => None,
      ContextEnum::AnytimeBounds(..) => None,
      ContextEnum::DiffMinMaxProb(_) => None,
      ContextEnum::DiffAddMultProb(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffNandMultProb(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffMaxMultProb(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffNandMinProb(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffSampleKProofs(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffTopKProofs(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffTopBottomKClauses(c) => Some(c.provenance_context().input_tags().into_vec()),
      ContextEnum::DiffTopKProofsDebug(c) => Some(c.provenance_context().input_tags().into_none_prepended_vec()),
      ContextEnum::Custom(_) => None,
    }
  }

  /// Set the hyper-parameter `k`
  fn set_k(&mut self, k: usize) {
    match &mut self.ctx {
      ContextEnum::Unit(_) => (),
      ContextEnum::Proofs(_) => (),
      ContextEnum::Tropical(_) => (),
      ContextEnum::MinMaxProb(_) => (),
      ContextEnum::AddMultProb(_) => (),
      ContextEnum::TopKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::TopBottomKClauses(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::AnytimeBounds(..) => (),
      ContextEnum::DiffMinMaxProb(_) => (),
      ContextEnum::DiffAddMultProb(_) => (),
      ContextEnum::DiffNandMultProb(_) => (),
      ContextEnum::DiffMaxMultProb(_) => (),
      ContextEnum::DiffNandMinProb(_) => (),
      ContextEnum::DiffSampleKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopKProofsDebug(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::Custom(_) => (),
    }
  }

  /// Declare a relation given a string of type declaration.
  ///
  /// # Example
  ///
  /// ```
  /// # ctx = Context::new("unit", 3, None).unwrap();
  /// ctx.add_relation("atom(usize, usize)", None, None).unwrap();
  /// ```
  #[pyo3(signature=(relation, load_csv=None, demand=None))]
  fn add_relation(
    &mut self,
    relation: &str,
    load_csv: Option<Bound<'_, PyAny>>,
    demand: Option<String>,
  ) -> Result<String, BindingError> {
    // Get the attributes
    let mut attrs = Vec::new();

    if let Some(obj) = load_csv {
      let csv_file = obj
        .extract::<CSVFileOptions>()
        .or_else(|_| obj.extract::<String>().map(CSVFileOptions::new))
        .map_err(|_| BindingError::InvalidLoadCSVArg)?;
      attrs.push(csv_file.into());
    }

    // Demand attribute
    if let Some(d) = demand {
      attrs.push(Attribute {
        name: "demand".to_string(),
        args: vec![AttributeArgument::string(d)],
      })
    }

    // Add relation
    let rtd = match_context!(&mut self.ctx, c, c.add_relation_with_attributes(relation, attrs)?);

    // Return; unwrap is ok here since we made sure that the relation is successfully added and there
    // will always be only one relation being added
    Ok(rtd.predicates().next().unwrap().to_string())
  }

  /// Add a list of facts to a relation
  fn add_facts(&mut self, relation: &str, elems: &Bound<'_, PyList>) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, add_py_facts(c, relation, elems))
  }

  fn add_joint_choice(&mut self, outcomes: Vec<PyJointOutcome<'_>>) -> Result<(), BindingError> {
    let ContextEnum::AnytimeBounds(ctx, choices) = &mut self.ctx else {
      return Err(PyValueError::new_err("add_joint_choice requires the anytime_bounds mode").into());
    };
    let env = ctx.runtime_environment().into();
    let choice = outcomes
      .into_iter()
      .map(|(weight, relations)| {
        let probability = extract_diff_probability(&weight)?;
        if !(0.0..=1.0).contains(&probability) {
          return Err(py_bounds_error(anytime_bounds::BoundsError::InvalidProbability(probability)).into());
        }
        let mut rows = Vec::new();
        for (name, tuples) in relations {
          let ty = ctx
            .relation_type(&name)
            .ok_or_else(|| BindingError::UnknownRelation(name.clone()))?;
          for tuple in tuples {
            rows.push((name.clone(), from_python_tuple(&tuple, &ty, &env)?));
          }
        }
        Ok(JointOutcome {
          weight: InputExclusiveDiffProb::new(probability, (&weight).into(), None),
          rows,
        })
      })
      .collect::<Result<Vec<_>, BindingError>>()?;
    if choice.iter().all(|outcome| outcome.rows.is_empty()) {
      return Err(PyValueError::new_err("a joint choice must contain at least one fact").into());
    }
    choices.push(choice);
    Ok(())
  }

  /// Add a rule
  #[pyo3(signature=(rule, tag=None, demand=None))]
  fn add_rule(
    &mut self,
    rule: &str,
    tag: Option<Bound<'_, PyAny>>,
    demand: Option<String>,
  ) -> Result<(), BindingError> {
    // Attributes
    let mut attrs = Vec::new();

    // Demand attribute
    if let Some(d) = demand {
      attrs.push(Attribute {
        name: "demand".to_string(),
        args: vec![AttributeArgument::string(d)],
      })
    }

    match_context!(&mut self.ctx, c, add_py_rule(c, rule, tag, attrs))
  }

  /// Register a foreign function
  fn register_foreign_function(&mut self, f: Py<PyAny>) -> Result<(), BindingError> {
    let ff = PythonForeignFunction::new(f);
    match_context!(&mut self.ctx, c, c.register_foreign_function(ff)?);
    Ok(())
  }

  /// Register a foreign predicate
  fn register_foreign_predicate(&mut self, f: Py<PyAny>) -> Result<(), BindingError> {
    let fp = PythonForeignPredicate::new(f);
    match_context!(&mut self.ctx, c, c.register_foreign_predicate(fp)?);
    Ok(())
  }

  /// Register a foreign attribute
  fn register_foreign_attribute(&mut self, attr: Py<PyAny>) -> Result<(), BindingError> {
    let py_attr = PythonForeignAttribute::new(attr);
    match_context!(&mut self.ctx, c, c.register_foreign_attribute(py_attr)?);
    Ok(())
  }

  /// Execute the program
  ///
  /// If the context's ram program is already compiled, the program will be directly executed.
  /// Otherwise, the ram program will be compiled and than be executed.
  fn run(&mut self) -> Result<(), BindingError> {
    self.require_standard_evaluation()?;
    match_context!(&mut self.ctx, c, c.run()?);
    Ok(())
  }

  /// Execute the program while monitoring the provenance and tags
  fn run_with_debug_tag(&mut self) -> Result<(), BindingError> {
    self.require_standard_evaluation()?;
    let m = monitor::DebugTagsMonitor;
    match_context!(&mut self.ctx, c, c.run_with_monitor(&m)?);
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  #[pyo3(signature=(relation, tuple, tolerance, max_facts, normalize_disjunctions=false, fact_order="input", checkpoint_schedule="doubling", *, second_order=false, gradient_targets=None))]
  fn evaluate_bounds(
    &mut self,
    relation: &str,
    tuple: &Bound<'_, PyTuple>,
    tolerance: Option<f64>,
    max_facts: usize,
    normalize_disjunctions: bool,
    fact_order: &str,
    checkpoint_schedule: &str,
    second_order: bool,
    gradient_targets: Option<Vec<(Bound<'_, PyAny>, f64)>>,
  ) -> Result<PyBoundsResult, BindingError> {
    let (ctx, joint) = self.bounds_context()?;
    let options = bounds_options(
      tolerance,
      max_facts,
      normalize_disjunctions,
      fact_order,
      checkpoint_schedule,
    )?;
    ctx.compile()?;
    let py = tuple.py();
    let tuple = bounds_query_tuple(ctx, relation, tuple)?;
    let (inputs, tensors) = snapshot_bounds_inputs(
      ctx,
      joint,
      py,
      tolerance.is_none() || gradient_targets.is_some(),
      std::iter::empty(),
    )?;
    let targets = gradient_targets
      .map(|targets| bounds_gradient_targets(py, &tensors, targets))
      .transpose()?;
    let program = ctx.ram();
    let result = py
      .detach(|| {
        if let Some(targets) = targets {
          anytime_bounds::evaluate_until_gradients(program, inputs, relation, &tuple, options, &targets, second_order)
        } else if second_order {
          anytime_bounds::evaluate_with_hessian(program, inputs, relation, &tuple, options)
        } else {
          anytime_bounds::evaluate(program, inputs, relation, &tuple, options)
        }
      })
      .map_err(py_bounds_error)?;
    Ok(py_bounds_result(py, result, &tensors))
  }

  fn start_bounds(
    &mut self,
    relation: &str,
    tuple: &Bound<'_, PyTuple>,
    tolerance: f64,
    normalize_disjunctions: bool,
    fact_order: &str,
    checkpoint_schedule: &str,
  ) -> Result<BoundsSession, BindingError> {
    let (ctx, joint) = self.bounds_context()?;
    let options = bounds_options(
      Some(tolerance),
      0,
      normalize_disjunctions,
      fact_order,
      checkpoint_schedule,
    )?;
    ctx.compile()?;
    let py = tuple.py();
    let tuple = bounds_query_tuple(ctx, relation, tuple)?;
    let (inputs, _) = snapshot_bounds_inputs(ctx, joint, py, false, std::iter::empty())?;
    let program = ctx.ram();
    let session = py
      .detach(|| {
        anytime_bounds::BoundsSession::new(
          program,
          inputs,
          (relation.to_owned(), tuple),
          tolerance,
          options.normalize_disjunctions,
          options.fact_order,
          options.checkpoint_schedule,
        )
      })
      .map_err(py_bounds_error)?;
    Ok(BoundsSession(session))
  }

  #[allow(clippy::too_many_arguments)]
  fn evaluate_relation_bounds(
    &mut self,
    py: Python<'_>,
    relation: &str,
    tolerance: f64,
    max_facts: usize,
    normalize_disjunctions: bool,
    fact_order: &str,
    checkpoint_schedule: &str,
  ) -> Result<Vec<(Py<PyAny>, PyBoundsInterval)>, BindingError> {
    let options = bounds_options(
      Some(tolerance),
      max_facts,
      normalize_disjunctions,
      fact_order,
      checkpoint_schedule,
    )?;
    self.bounds_relation_rows(py, relation, options, |result, _| py_bounds_interval(&result))
  }

  fn evaluate_relation_bounds_with_gradients(
    &mut self,
    py: Python<'_>,
    relation: &str,
    max_facts: usize,
    normalize_disjunctions: bool,
    fact_order: &str,
  ) -> Result<Vec<(Py<PyAny>, PyBoundsResult)>, BindingError> {
    let options = bounds_options(None, max_facts, normalize_disjunctions, fact_order, "doubling")?;
    self.bounds_relation_rows(py, relation, options, |result, tensors| {
      py_bounds_result(py, result, tensors)
    })
  }

  #[allow(clippy::too_many_arguments)]
  fn evaluate_bounds_batch(
    &mut self,
    py: Python<'_>,
    queries: Vec<(String, Bound<'_, PyTuple>)>,
    inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
    tolerance: Option<f64>,
    max_facts: usize,
    normalize_disjunctions: bool,
    fact_order: &str,
    checkpoint_schedule: &str,
  ) -> Result<Vec<PyBoundsResult>, BindingError> {
    let (ctx, joint) = self.bounds_context()?;
    let options = bounds_options(
      tolerance,
      max_facts,
      normalize_disjunctions,
      fact_order,
      checkpoint_schedule,
    )?;
    if inputs.values().any(|examples| examples.len() != queries.len()) {
      return Err(BindingError::InvalidBatchSize);
    }
    if queries.is_empty() {
      return Ok(Vec::new());
    }
    ctx.compile()?;
    let queries = queries
      .into_iter()
      .map(|(relation, tuple)| bounds_query_tuple(ctx, &relation, &tuple).map(|tuple| (relation, tuple)))
      .collect::<Result<Vec<_>, _>>()?;
    let extra = process_batched_inputs::<diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, AF>, _>(
      inputs,
      ctx.runtime_environment(),
      |relation| ctx.relation_type(relation),
    )?;
    let snapshot_count = if extra.is_empty() { 1 } else { queries.len() };
    let (snapshots, tensors): (Vec<_>, Vec<_>) = (0..snapshot_count)
      .map(|index| {
        let facts = extra
          .iter()
          .flat_map(|(relation, examples)| examples[index].iter().map(move |fact| (relation.as_str(), fact)));
        snapshot_bounds_inputs(ctx, joint, py, tolerance.is_none(), facts)
      })
      .collect::<PyResult<Vec<_>>>()?
      .into_iter()
      .unzip();
    let program = ctx.ram();
    let results = py
      .detach(|| {
        if snapshot_count == 1 {
          return anytime_bounds::evaluate_batch(program, &snapshots[0], &queries, options);
        }
        queries
          .iter()
          .enumerate()
          .map(|(index, (relation, tuple))| {
            let facts = snapshots[index].iter().cloned();
            anytime_bounds::evaluate(program, facts, relation, tuple, options)
          })
          .collect::<Result<Vec<_>, _>>()
      })
      .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(
      results
        .into_iter()
        .enumerate()
        .map(|(index, result)| py_bounds_result(py, result, &tensors[if snapshot_count == 1 { 0 } else { index }]))
        .collect(),
    )
  }

  /// Check if the tuple matches the type of the queried relation.
  ///
  /// Error will be returned if the relation does not exist.
  fn check_tuple(&self, relation: &str, tup: &Bound<'_, PyAny>) -> Result<bool, BindingError> {
    match_context!(&self.ctx, c, check_py_tuple(c, relation, tup))
  }

  /// Check if a set of tuples match the type of the queried relation.
  ///
  /// Error will be returned if the relation does not exist.
  fn check_tuples(&self, relation: &str, tups: Vec<Bound<'_, PyAny>>) -> Result<bool, BindingError> {
    match_context!(&self.ctx, c, check_py_tuples(c, relation, tups))
  }

  /// Run a batch of tasks
  ///
  /// # Arguments
  ///
  /// * `iter_limit` - the optional iteration count limit. If `None`, there is no limit to iteration count.
  /// * `output_relations` - a doubly nested vector of strings.
  ///   The first level is an array of batched tasks, and the second level is the list of output relations for that particular task
  /// * `inputs` - the input facts, a mapping from the relation name to a batch of input facts.
  ///   For each task, there is a `PyList` of input facts and another optional set of annotated disjunctions
  /// * `parallel` - whether the batch is to be executed in parallel
  fn run_batch(
    &self,
    output_relations: Vec<Vec<String>>,
    inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
    parallel: bool,
  ) -> Result<Vec<Vec<Collection>>, BindingError> {
    self.require_standard_evaluation()?;
    // Sanity check: has input
    if inputs.is_empty() {
      return Err(BindingError::EmptyBatchInput);
    }

    // Sanity check: all input facts share the same batch size
    if !is_all_equal(inputs.values().map(|batch| batch.len())) {
      return Err(BindingError::InvalidBatchSize);
    }

    // Sanity check: non-empty batch size
    if inputs.iter().next().unwrap().1.is_empty() {
      return Err(BindingError::InvalidBatchSize);
    }

    // Depending on whether parallelism is enabled, choose the subsequent function to call
    if parallel {
      self.run_batch_parallel(output_relations, inputs)
    } else {
      self.run_batch_serial(output_relations, inputs)
    }
  }

  /// Get the output collection of the relation
  ///
  /// Has to be called after `ctx.run()`, otherwise the output collection will not be computed.
  /// Error is returned if the relation is not computed or does not exist.
  fn relation(&mut self, r: &str) -> Result<Collection, BindingError> {
    let (maybe_coll_enum, env) = match_context!(&mut self.ctx, c, {
      (get_output_collection(c, r), c.runtime_environment())
    });
    maybe_coll_enum.map(|c| c.into_collection(env))
  }

  /// Get the output collection of the relation, while monitoring the provenance and tags
  ///
  /// Has to be called after `ctx.run()`, otherwise the output collection will not be computed.
  /// Error is returned if the relation is not computed or does not exist.
  fn relation_with_debug_tag(&mut self, r: &str) -> Result<Collection, BindingError> {
    let m = monitor::DebugTagsMonitor;
    let (maybe_coll_enum, env) = match_context!(&mut self.ctx, c, {
      (get_output_collection_monitor(c, &m, r), c.runtime_environment())
    });
    maybe_coll_enum.map(|c| c.into_collection(env))
  }

  /// Check if the context contains a relation
  fn has_relation(&self, r: &str) -> bool {
    match_context!(&self.ctx, c, c.has_relation(r))
  }

  /// Check if the relation is computed in this context
  fn relation_is_computed(&self, r: &str) -> bool {
    match_context!(&self.ctx, c, c.is_computed(r))
  }

  /// Get the number of relations in this context.
  /// If `include_hidden` is set `true`, the result will also count the hidden relations
  #[pyo3(signature=(include_hidden = false))]
  fn num_relations(&self, include_hidden: bool) -> usize {
    if include_hidden {
      match_context!(&self.ctx, c, c.num_all_relations())
    } else {
      match_context!(&self.ctx, c, c.num_relations())
    }
  }

  /// Get a list of relations in this context.
  /// If `include_hidden` is set `true`, the result will also include the hidden relations
  #[pyo3(signature=(include_hidden = false))]
  fn relations(&self, include_hidden: bool) -> Vec<String> {
    if include_hidden {
      match_context!(&self.ctx, c, c.all_relations())
    } else {
      match_context!(&self.ctx, c, c.relations())
    }
  }

  fn relation_field_names(&self, relation: &str) -> Option<Vec<Option<String>>> {
    match_context!(&self.ctx, c, c.relation_field_names(relation).cloned())
  }
}

impl Context {
  fn bounds_context(&mut self) -> Result<(&mut DiffWeightedContext, &[Vec<JointOutcome>]), BindingError> {
    match &mut self.ctx {
      ContextEnum::AnytimeBounds(ctx, choices) => Ok((ctx, choices)),
      _ => Err(PyValueError::new_err("evaluate_bounds requires the anytime_bounds mode").into()),
    }
  }

  fn bounds_relation_rows<T>(
    &mut self,
    py: Python<'_>,
    relation: &str,
    options: anytime_bounds::BoundsOptions,
    convert: impl Fn(anytime_bounds::BoundsResult, &BoundsTensors) -> T,
  ) -> Result<Vec<(Py<PyAny>, T)>, BindingError> {
    let (ctx, joint) = self.bounds_context()?;
    ctx.compile()?;
    let (inputs, tensors) = snapshot_bounds_inputs(ctx, joint, py, options.tolerance.is_none(), std::iter::empty())?;
    let program = ctx.ram();
    let rows = py
      .detach(|| anytime_bounds::evaluate_relation(program, inputs, relation, options))
      .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let env = ctx.runtime_environment().into();
    rows
      .into_iter()
      .map(|(tuple, result)| {
        let tuple = to_python_tuple(&tuple, &env)
          .ok_or_else(|| PyValueError::new_err(format!("cannot convert a bounds tuple from relation `{relation}`")))?;
        Ok((tuple, convert(result, &tensors)))
      })
      .collect()
  }

  fn require_standard_evaluation(&self) -> Result<(), BindingError> {
    if matches!(self.ctx, ContextEnum::AnytimeBounds(..)) {
      Err(bounds_operation_error())
    } else {
      Ok(())
    }
  }

  fn run_batch_parallel(
    &self,
    output_relations: Vec<Vec<String>>,
    inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
  ) -> Result<Vec<Vec<Collection>>, BindingError> {
    // Helper function for running batch parallel
    fn run<C>(
      c: &IntegrateContext<C, AF>,
      output_relations: Vec<Vec<String>>,
      inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
    ) -> Result<Vec<Vec<Collection>>, BindingError>
    where
      C: PythonProvenance + Clone + std::marker::Sync,
      <C as Provenance>::InputTag: std::marker::Sync,
      <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
      <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();
      let inputs = process_batched_inputs::<C, _>(inputs, c.runtime_environment(), |r| c.relation_type(r))?;
      run_batch_parallel(c, batch_size, inputs, output_relations)
    }

    // Run the helper for each context
    match_context!(&self.ctx, c, run(c, output_relations, inputs))
  }

  fn run_batch_serial(
    &self,
    output_relation: Vec<Vec<String>>,
    inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
  ) -> Result<Vec<Vec<Collection>>, BindingError> {
    let batch_size = inputs.iter().next().unwrap().1.len();
    (0..batch_size)
      .map(|i| {
        let mut temp_ctx = self.clone();
        for (relation, relation_inputs) in &inputs {
          temp_ctx.add_facts(relation, &relation_inputs[i])?;
        }
        temp_ctx.run()?;
        output_relation[i].iter().map(|r| temp_ctx.relation(r)).collect()
      })
      .collect()
  }
}

fn bounds_operation_error() -> BindingError {
  PyValueError::new_err(
    "use evaluate_bounds, evaluate_bounds_batch, or evaluate_relation_bounds in the anytime_bounds mode",
  )
  .into()
}

fn py_bounds_error(error: anytime_bounds::BoundsError) -> PyErr {
  PyValueError::new_err(error.to_string())
}

fn bounds_options(
  tolerance: Option<f64>,
  max_facts: usize,
  normalize_disjunctions: bool,
  fact_order: &str,
  checkpoint_schedule: &str,
) -> PyResult<anytime_bounds::BoundsOptions> {
  let fact_order = match fact_order {
    "input" => anytime_bounds::FactOrder::Input,
    "weight" => anytime_bounds::FactOrder::Weight,
    "required" => anytime_bounds::FactOrder::Required,
    "lookahead" => anytime_bounds::FactOrder::Lookahead,
    _ => {
      return Err(PyValueError::new_err(
        "fact_order must be 'input', 'weight', 'required', or 'lookahead'",
      ))
    }
  };
  let checkpoint_schedule = match checkpoint_schedule {
    "doubling" => anytime_bounds::CheckpointSchedule::Doubling,
    "adaptive" => anytime_bounds::CheckpointSchedule::Adaptive,
    _ => {
      return Err(PyValueError::new_err(
        "checkpoint_schedule must be 'doubling' or 'adaptive'",
      ))
    }
  };
  Ok(anytime_bounds::BoundsOptions {
    tolerance,
    max_facts,
    normalize_disjunctions,
    fact_order,
    checkpoint_schedule,
  })
}

fn bounds_query_tuple(
  ctx: &DiffWeightedContext,
  relation: &str,
  tuple: &Bound<'_, PyTuple>,
) -> Result<Tuple, BindingError> {
  let tuple_type = ctx
    .ram()
    .relation_tuple_type(relation)
    .ok_or_else(|| BindingError::UnknownRelation(relation.into()))?;
  Ok(from_python_tuple(
    tuple.as_any(),
    &tuple_type,
    &ctx.runtime_environment().into(),
  )?)
}

type BoundsTensors = HashMap<usize, Py<PyAny>>;

fn bounds_gradient_targets(
  py: Python<'_>,
  tensors: &BoundsTensors,
  targets: Vec<(Bound<'_, PyAny>, f64)>,
) -> PyResult<Vec<anytime_bounds::GradientTarget>> {
  let mut selected = Vec::new();
  for (target, tolerance) in targets {
    let before = selected.len();
    selected.extend(tensors.iter().filter_map(|(&input, tensor)| {
      target
        .is(tensor.bind(py))
        .then_some(anytime_bounds::GradientTarget { input, tolerance })
    }));
    if selected.len() == before {
      return Err(PyValueError::new_err(
        "gradient target is not a trainable fact weight in this context",
      ));
    }
  }
  Ok(selected)
}

fn snapshot_bounds_inputs<'a>(
  ctx: &mut DiffWeightedContext,
  joint: &[Vec<JointOutcome>],
  py: Python<'_>,
  gradients: bool,
  extra: impl IntoIterator<Item = (&'a str, &'a BoundsFact)>,
) -> PyResult<(Vec<anytime_bounds::BoundsInput>, BoundsTensors)> {
  let modules = py.import("sys")?.getattr("modules")?;
  let torch = modules.cast::<PyDict>().map_err(PyErr::from)?.get_item("torch")?;
  let tensor_type = torch
    .filter(|module| !module.is_none())
    .map(|module| module.getattr("Tensor"))
    .transpose()?;
  let mut relations = BTreeMap::<&str, (&[BoundsFact], Vec<&BoundsFact>)>::new();
  for (name, relation) in &ctx.edb().extensional_relations {
    relations.entry(name).or_default().0 = relation.pending_static_facts();
  }
  for (name, fact) in extra {
    relations.entry(name).or_default().1.push(fact);
  }
  let mut inputs = Vec::new();
  let mut tensors = HashMap::new();
  let mut push = |rows, tag: Option<&InputExclusiveDiffProb<ExtTag>>, group| {
    let mut probability = tag.map_or(1.0, |tag| tag.prob);
    let mut differentiate = false;
    if let (Some(tensor_type), Some(source)) = (
      &tensor_type,
      tag.and_then(|tag| tag.external_tag.as_ref()).into_option(),
    ) {
      let tensor = source.bind(py);
      if tensor.is_instance(tensor_type)? {
        if !tensor.call_method0("is_floating_point")?.is_truthy()? {
          return Err(PyValueError::new_err(
            "bounds probability tensors must have a floating-point dtype",
          ));
        }
        probability = tensor.call_method0("item")?.extract()?;
        differentiate = gradients && tensor.getattr("requires_grad")?.is_truthy()?;
        if differentiate {
          tensors.insert(inputs.len(), source);
        }
      }
    }
    inputs.push(anytime_bounds::BoundsInput {
      rows,
      probability,
      group,
      differentiate,
    });
    Ok(())
  };
  for (name, (facts, extra)) in relations {
    for (tag, fact_tuple) in facts.iter().chain(extra) {
      let tag = tag.as_ref();
      push(
        anytime_bounds::BoundsRows::Fact((name.into(), fact_tuple.clone())),
        tag,
        tag
          .and_then(|tag| tag.exclusion)
          .map(anytime_bounds::BoundsInputGroup::Categorical),
      )?;
    }
  }
  for (group, outcomes) in joint.iter().enumerate() {
    for outcome in outcomes {
      push(
        anytime_bounds::BoundsRows::Joint(outcome.rows.clone()),
        Some(&outcome.weight),
        Some(anytime_bounds::BoundsInputGroup::Joint(group)),
      )?;
    }
  }
  Ok((inputs, tensors))
}

fn py_bounds_interval(result: &anytime_bounds::BoundsResult) -> PyBoundsInterval {
  let reason = match result.stop_reason {
    anytime_bounds::BoundsStop::Tolerance => "tolerance",
    anytime_bounds::BoundsStop::FactLimit => "fact_limit",
    anytime_bounds::BoundsStop::TimeLimit => "time_limit",
    anytime_bounds::BoundsStop::Complete => "complete",
  };
  (
    result.interval.lower,
    result.interval.upper,
    result.active_facts,
    reason,
  )
}

fn py_bounds_result(py: Python<'_>, result: anytime_bounds::BoundsResult, tensors: &BoundsTensors) -> PyBoundsResult {
  let interval = py_bounds_interval(&result);
  let gradients = result
    .gradients
    .into_iter()
    .map(|gradient| {
      (
        tensors[&gradient.input].clone_ref(py),
        gradient.lower,
        gradient.upper,
        gradient.exact,
      )
    })
    .collect();
  (interval, result.exact_probability, gradients, result.hessian)
}

fn add_py_rule<P>(
  c: &mut IntegrateContext<P, AF>,
  rule: &str,
  tag: Option<Bound<'_, PyAny>>,
  attrs: Vec<Attribute>,
) -> Result<(), BindingError>
where
  P: PythonProvenance,
{
  let tag = from_optional_python_tag(tag.as_ref(), P::process_py_tag)?;
  c.add_rule_with_options(rule, tag, attrs)?;
  Ok(())
}

fn add_py_facts<P>(
  c: &mut IntegrateContext<P, AF>,
  relation: &str,
  elems: &Bound<'_, PyList>,
) -> Result<(), BindingError>
where
  P: PythonProvenance,
{
  if let Some(tuple_type) = c.relation_type(relation) {
    let tuples = P::process_typed_py_facts(elems, &tuple_type, c.runtime_environment())?;
    c.add_facts(relation, tuples, false)?;
    Ok(())
  } else {
    Err(BindingError::UnknownRelation(relation.to_string()))
  }
}

fn check_py_tuple<P>(
  c: &IntegrateContext<P, AF>,
  relation: &str,
  py_tup: &Bound<'_, PyAny>,
) -> Result<bool, BindingError>
where
  P: PythonProvenance,
{
  if let Some(tuple_type) = c.relation_type(relation) {
    Ok(from_python_tuple(py_tup, &tuple_type, &c.runtime_environment().into()).is_ok())
  } else {
    Err(BindingError::UnknownRelation(relation.to_string()))
  }
}

fn check_py_tuples<C>(
  c: &IntegrateContext<C, AF>,
  relation: &str,
  py_tups: Vec<Bound<'_, PyAny>>,
) -> Result<bool, BindingError>
where
  C: PythonProvenance,
{
  if let Some(tuple_type) = c.relation_type(relation) {
    for py_tup in py_tups {
      if from_python_tuple(&py_tup, &tuple_type, &c.runtime_environment().into()).is_err() {
        return Ok(false);
      }
    }
    Ok(true)
  } else {
    Err(BindingError::UnknownRelation(relation.to_string()))
  }
}

type BatchedRelationInputs<Tag> = Vec<(String, Vec<Vec<(Option<Tag>, Tuple)>>)>;

fn process_batched_inputs<P, G>(
  inputs: HashMap<String, Vec<Bound<'_, PyList>>>,
  env: &RuntimeEnvironment,
  get_relation_type: G,
) -> PyResult<BatchedRelationInputs<P::InputTag>>
where
  P: PythonProvenance,
  G: Fn(&str) -> Option<TupleType>,
{
  inputs
    .into_iter()
    .map(|(r, b)| {
      let tuple_type = get_relation_type(&r).ok_or(BindingError::UnknownRelation(r.clone()))?;
      let batch = b
        .into_iter()
        .map(|elems| P::process_typed_py_facts(&elems, &tuple_type, env))
        .collect::<PyResult<Vec<_>>>()?;
      Ok((r, batch))
    })
    .collect::<PyResult<Vec<_>>>()
}

fn run_batch_parallel<C>(
  integrate_context: &IntegrateContext<C, AF>,
  batch_size: usize,
  inputs: BatchedRelationInputs<C::InputTag>,
  batch_output_relations: Vec<Vec<String>>,
) -> Result<Vec<Vec<Collection>>, BindingError>
where
  C: PythonProvenance + Clone + std::marker::Sync,
  <C as Provenance>::InputTag: std::marker::Sync,
  <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
  <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
{
  let internal = integrate_context.internal_context();
  // Release the caller's GIL while workers may need to invoke Python callbacks.
  Python::attach(|py| {
    py.detach(|| {
      (0..batch_size)
        .into_par_iter()
        .map(|i| {
          let output_relations = &batch_output_relations[i];
          let mut temp_ctx = internal.clone();
          for (relation, batch) in &inputs {
            temp_ctx.add_facts(relation, batch[i].clone(), false)?;
          }
          temp_ctx.run()?;
          output_relations
            .iter()
            .map(|r| {
              let computed = temp_ctx
                .computed_relation(r)
                .ok_or(BindingError::RelationNotComputed(r.to_string()))?;
              let collection_enum = C::to_collection_enum(computed, temp_ctx.provenance_context());
              let collection = collection_enum.into_collection(&temp_ctx.runtime_env);
              Ok(collection)
            })
            .collect::<Result<Vec<Collection>, _>>()
        })
        .collect()
    })
  })
}

fn is_all_equal<T: Iterator<Item = usize> + Clone>(i: T) -> bool {
  i.clone().min() == i.max()
}

fn get_output_collection<C>(
  c: &mut IntegrateContext<C, ArcFamily>,
  r: &str,
) -> Result<CollectionEnum<ArcFamily>, BindingError>
where
  C: PythonProvenance,
{
  if c.has_relation(r) {
    let collection = c
      .computed_relation(r)
      .ok_or_else(|| BindingError::RelationNotComputed(r.to_string()))?;
    Ok(C::to_collection_enum(collection, c.provenance_context()))
  } else {
    Err(BindingError::UnknownRelation(r.to_string()))
  }
}

fn get_output_collection_monitor<C, M>(
  c: &mut IntegrateContext<C, ArcFamily>,
  m: &M,
  r: &str,
) -> Result<CollectionEnum<ArcFamily>, BindingError>
where
  C: PythonProvenance,
  M: monitor::Monitor<C>,
{
  if c.has_relation(r) {
    let collection = c
      .computed_relation_with_monitor(r, m)
      .ok_or_else(|| BindingError::RelationNotComputed(r.to_string()))?;
    Ok(C::to_collection_enum(collection, c.provenance_context()))
  } else {
    Err(BindingError::UnknownRelation(r.to_string()))
  }
}
