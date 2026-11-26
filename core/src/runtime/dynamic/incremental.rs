use std::collections::*;

use super::*;
use crate::common::tuple::*;
use crate::compiler::ram;
use crate::runtime::database::extensional::*;
use crate::runtime::database::intentional::*;
use crate::runtime::database::*;
use crate::runtime::env::*;
use crate::runtime::error::*;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;
use crate::utils::*;

#[derive(Clone, Debug)]
pub struct ExecutionOptions {
  pub type_check: bool,
  /// Retain results for subsequent runs in the same runtime and provenance context.
  pub incremental_maintain: bool,
  pub retain_internal_when_recover: bool,
}

impl Default for ExecutionOptions {
  fn default() -> Self {
    Self {
      type_check: true,
      incremental_maintain: false,
      retain_internal_when_recover: true,
    }
  }
}

struct StratumExecution<'a, Prov> {
  ram_program: &'a ram::Program,
  strata_info: &'a StrataInformation,
  runtime: &'a RuntimeEnvironment,
  ctx: &'a Prov,
  interrupted: Option<&'a dyn Fn() -> bool>,
}

pub struct DynamicExecutionContext<Prov: Provenance, Ptr: PointerFamily = RcFamily> {
  pub options: ExecutionOptions,
  pub program: ram::Program,
  pub edb: ExtensionalDatabase<Prov>,
  pub idb: IntentionalDatabase<Prov, Ptr>,
  cached_early_discard: Option<bool>,
}

impl<Prov: Provenance, Ptr: PointerFamily> Clone for DynamicExecutionContext<Prov, Ptr> {
  fn clone(&self) -> Self {
    Self {
      options: self.options.clone(),
      program: self.program.clone(),
      edb: self.edb.clone(),
      idb: self.idb.clone(),
      cached_early_discard: self.cached_early_discard,
    }
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> DynamicExecutionContext<Prov, Ptr> {
  pub fn new() -> Self {
    let options = ExecutionOptions::default();
    Self::new_with_options(options)
  }

  pub fn new_with_options(options: ExecutionOptions) -> Self {
    Self::new_with_program_and_options(ram::Program::new(), options)
  }

  pub fn new_with_program_and_options(program: ram::Program, options: ExecutionOptions) -> Self {
    let edb = ExtensionalDatabase::with_relation_types_and_options(program.relation_types(), options.type_check);
    let idb = IntentionalDatabase::default();
    Self {
      options,
      program,
      edb,
      idb,
      cached_early_discard: None,
    }
  }

  pub fn clone_with_new_provenance<Prov2: Provenance>(&self) -> DynamicExecutionContext<Prov2, Ptr>
  where
    Prov2::InputTag: ConvertFromInputTag<Prov::InputTag>,
  {
    DynamicExecutionContext {
      options: self.options.clone(),
      program: self.program.clone(),
      edb: self.edb.clone_with_new_provenance::<Prov2>(),
      idb: self.idb.clone_with_new_provenance::<Prov2>(),
      cached_early_discard: None,
    }
  }

  pub fn set_non_incremental(&mut self) {
    self.options.incremental_maintain = false;
    self.cached_early_discard = None;
  }

  /// Directly execute the program stored in the file
  pub fn execute(&mut self, runtime: &RuntimeEnvironment, ctx: &mut Prov) -> Result<(), RuntimeError> {
    self.execute_inner::<()>(None, runtime, ctx, None, None)
  }

  pub(crate) fn execute_interruptible(
    &mut self,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
    interrupted: &dyn Fn() -> bool,
  ) -> Result<(), RuntimeError> {
    self.execute_inner::<()>(None, runtime, ctx, None, Some(interrupted))
  }

  pub fn incremental_execute(
    &mut self,
    program: ram::Program,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> Result<(), RuntimeError> {
    self.execute_inner::<()>(Some(program), runtime, ctx, None, None)
  }

  /// Directly execute the program stored in the file
  pub fn execute_with_monitor<M>(&mut self, runtime: &RuntimeEnvironment, ctx: &Prov, m: &M) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    self.execute_inner(None, runtime, ctx, Some(m), None)?;
    m.observe_finish_execution();
    Ok(())
  }

  pub fn incremental_execute_with_monitor<M>(
    &mut self,
    program: ram::Program,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
    m: &M,
  ) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    self.execute_inner(Some(program), runtime, ctx, Some(m), None)
  }

  fn execute_inner<M: Monitor<Prov>>(
    &mut self,
    maybe_new_program: Option<ram::Program>,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
    monitor: Option<&M>,
    interrupted: Option<&dyn Fn() -> bool>,
  ) -> Result<(), RuntimeError> {
    check_interruption(interrupted)?;
    let program_ref = maybe_new_program.as_ref().unwrap_or(&self.program);
    self.edb.populate_program_facts(runtime, program_ref)?;

    let can_cache = self.options.incremental_maintain
      && self.options.retain_internal_when_recover
      && supports_result_reuse(program_ref, runtime);
    let previous_cache = self.cached_early_discard.take();
    let mut incremental_result = std::mem::take(&mut self.idb);
    if can_cache && previous_cache == Some(runtime.early_discard) {
      let mut need_update = self.edb.need_update_relations();
      need_update.extend(program_ref.relations().filter_map(|relation| {
        let predicate = &relation.predicate;
        self
          .program
          .relation(predicate)
          .filter(|old| {
            old.immutable != relation.immutable
              || old.storage != relation.storage
              || old.scheduler != relation.scheduler
          })
          .map(|_| predicate.clone())
      }));
      let reusable = self
        .program
        .persistent_relations_with_need_update_relations(program_ref, &need_update);
      incremental_result.retain_relations(&reusable);
    } else if self.options.incremental_maintain || maybe_new_program.is_some() {
      incremental_result = IntentionalDatabase::default();
    }

    // Internalize EDB relations
    if let Some(m) = monitor {
      self.edb.internalize_with_monitor(runtime, ctx, m);
    } else {
      self.edb.internalize(runtime, ctx);
    }

    // Generate stratum information
    let strata_info = stratum_inputs_outputs(program_ref);

    // Make sure that all immutable relations are existed in the edb
    for relation in program_ref.relations() {
      if relation.immutable {
        self.edb.tap_relation(&relation.predicate);
      }
    }

    let execution = StratumExecution {
      ram_program: program_ref,
      strata_info: &strata_info,
      runtime,
      ctx,
      interrupted,
    };
    // Go through each stratum
    for i in 0..program_ref.strata.len() {
      check_interruption(interrupted)?;
      // Run the stratum to get the result
      let result = self.execute_stratum(i, &incremental_result, &execution, monitor)?;

      // Extend the incremental result with the relations within their lifespan
      incremental_result.extend(result.into_iter().filter(|(name, _)| {
        // Check if the relation is a hidden one
        if program_ref.relation_unchecked(name).output.is_hidden() {
          // If it is hidden, additionally check if it is going to be used later
          if let Some(last_stratum) = strata_info.relation_lifespan.get(name) {
            // If the last stratum that this relation is used is later than current, then keep; otherwise drop
            return *last_stratum > i;
          }
        }
        true
      }));
    }

    // Store the result
    self.idb = incremental_result;

    // See if any EDB relations need to directly go into IDB
    for relation in program_ref.relations() {
      if relation.output.is_not_hidden() && !self.idb.has_relation(&relation.predicate) {
        if self.options.incremental_maintain {
          if let Some(edb_collection) = self.edb.get_dynamic_collection(&relation.predicate) {
            self
              .idb
              .insert_dynamic_collection(relation.predicate.clone(), edb_collection.clone_internal());
          }
        } else {
          if let Some(edb_collection) = self.edb.pop_dynamic_collection(&relation.predicate) {
            self
              .idb
              .insert_dynamic_collection(relation.predicate.clone(), edb_collection.clone());
          }
        }
      }
    }

    // Update the program
    if let Some(new_program) = maybe_new_program {
      self.program = new_program;
    }
    self.cached_early_discard = can_cache.then_some(runtime.early_discard);

    // Success!
    Ok(())
  }

  fn execute_stratum<M: Monitor<Prov>>(
    &self,
    stratum_id: usize,
    current_idb: &IntentionalDatabase<Prov, Ptr>,
    execution: &StratumExecution<'_, Prov>,
    monitor: Option<&M>,
  ) -> Result<IntentionalDatabase<Prov, Ptr>, RuntimeError> {
    let StratumExecution {
      ram_program,
      strata_info,
      runtime,
      ctx,
      interrupted,
    } = *execution;
    let stratum = &ram_program.strata[stratum_id];
    if let Some(m) = monitor {
      m.observe_executing_stratum(stratum_id);
    }
    let observe_load = |relation: &str, from_edb: bool| {
      if let Some(m) = monitor {
        m.observe_loading_relation(relation);
        if from_edb {
          m.observe_loading_relation_from_edb(relation);
        } else {
          m.observe_loading_relation_from_idb(relation);
        }
      }
    };

    let dyn_relas = stratum
      .relations
      .iter()
      .filter(|(r, _)| !current_idb.has_relation(r))
      .map(|(r, _)| r.clone())
      .collect::<HashSet<_>>();

    // Check if we need to compute anything new
    if dyn_relas.is_empty() {
      return Ok(IntentionalDatabase::new());
    }

    // Otherwise, do computation
    let mut iter = DynamicIteration::<Prov>::new();

    // Add input collections
    if self.options.incremental_maintain {
      for (rel, col) in &self.edb.extensional_relations {
        if ram_program.relation(rel).map(|r| r.immutable).unwrap_or(false) {
          observe_load(rel, true);
          iter.add_input_dynamic_collection(rel, col.internal.as_ref());
        }
      }
      for (rel, col) in current_idb {
        // A recovered immutable relation can have an output-only copy in the IDB.
        if !ram_program.relation(rel).is_some_and(|r| r.immutable) {
          observe_load(rel, false);
          iter.add_input_dynamic_collection(rel, col.internal_facts.as_ref());
        }
      }
    } else {
      // Non-incremental version:

      // First add dynamic collection for every immutable relations in the stratum
      for (predicate, relation) in &stratum.relations {
        if relation.immutable {
          observe_load(predicate, true);
          iter.add_input_dynamic_collection(predicate, self.edb.get_dynamic_collection(predicate).unwrap());
        }
      }

      // Then add dynamic collection for every input relations in the stratum
      if let Some(stratum_inputs) = strata_info.stratum_inputs.get(&stratum_id) {
        for (rel, _) in stratum_inputs {
          if ram_program.relation_unchecked(rel).immutable {
            observe_load(rel, true);
            // The collection could be immutable and thus will be from EDB
            iter.add_input_dynamic_collection(rel, self.edb.get_dynamic_collection(rel).unwrap());
          } else {
            observe_load(rel, false);
            // Otherwise it will be computed by previous stratum and thus from IDB
            iter.add_input_dynamic_collection(rel, current_idb.get_internal_collection(rel).unwrap());
          }
        }
      }
    }

    // Create dynamic relations; all of them will be in the output
    // Note: Unwrap is ok since the relation in stratum must be in the ram program
    for rela in dyn_relas
      .iter()
      .filter(|r| !ram_program.relation_unchecked(r).immutable)
    {
      iter.create_dynamic_relation(rela);

      // Check if we need it to be output
      if self.options.incremental_maintain
        || strata_info
          .stratum_outputs
          .get(&stratum_id)
          .is_some_and(|o| o.contains(rela))
        || ram_program.relation_unchecked(rela).output.is_not_hidden()
      {
        let storage = &ram_program.relation_unchecked(rela).storage;
        iter.add_output_relation(rela, storage);
      }

      // Check if we need it to be a goal
      if ram_program.relation(rela).unwrap().is_goal {
        iter.add_goal_relation(rela);
      }

      if let Some(scheduler) = &ram_program.relation_unchecked(rela).scheduler {
        iter.add_relation_scheduler(rela, scheduler.clone());
      }

      // Load external facts
      if let Some(facts) = self.edb.get_dynamic_collection(rela) {
        observe_load(rela, true);
        // Mutable relations need their EDB facts to go into dynamic relation
        let dataflow = dataflow::DynamicDataflow::dynamic_recent_collection(facts);
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dataflow_recent(ctx, &dataflow, runtime);
      }
    }

    // Add updates
    for update in &stratum.updates {
      if dyn_relas.contains(&update.target) {
        iter.add_update(update.clone());
      }
    }

    // Run!
    let result = iter.run_checked(ctx, runtime, monitor, interrupted)?;

    // Success!
    Ok(IntentionalDatabase::from_dynamic_collections(result.into_iter()))
  }

  pub fn add_facts(
    &mut self,
    relation: &str,
    facts: Vec<(Option<InputTagOf<Prov>>, Tuple)>,
  ) -> Result<(), DatabaseError> {
    // For incremental: when new fact is added, we need to recompute the relation
    if !facts.is_empty() {
      self.idb.remove_relation(relation);
    }

    // Add facts back
    self.edb.add_static_input_facts(relation, facts)
  }

  pub fn internal_relation(&self, r: &str) -> Option<DynamicCollectionRef<'_, Prov>> {
    self.idb.get_internal_collection(r)
  }

  pub fn recover(&mut self, r: &str, runtime: &RuntimeEnvironment, ctx: &Prov) {
    if !self.options.retain_internal_when_recover {
      self.cached_early_discard = None;
    }
    if self.idb.has_relation(r) {
      self
        .idb
        .recover(r, runtime, ctx, !self.options.retain_internal_when_recover);
    } else if self.edb.has_relation(r) {
      self
        .idb
        .recover_from_edb(r, runtime, ctx, &self.edb.extensional_relations[r]);
    }
  }

  pub fn recover_with_monitor<M: Monitor<Prov>>(&mut self, r: &str, runtime: &RuntimeEnvironment, ctx: &Prov, m: &M) {
    if !self.options.retain_internal_when_recover {
      self.cached_early_discard = None;
    }
    self
      .idb
      .recover_with_monitor(r, runtime, ctx, m, !self.options.retain_internal_when_recover)
  }

  pub fn relation_ref(&self, r: &str) -> Option<&DynamicOutputCollection<Prov>> {
    self.idb.get_output_collection_ref(r)
  }

  pub fn relation(&self, r: &str) -> Option<Ptr::Rc<DynamicOutputCollection<Prov>>> {
    self.idb.get_output_collection(r)
  }

  pub fn is_computed(&self, r: &str) -> bool {
    self.idb.has_relation(r)
  }

  pub fn num_relations(&self) -> usize {
    self.program.relation_to_stratum.len()
  }

  pub fn relations(&self) -> Vec<String> {
    self.program.relation_to_stratum.keys().cloned().collect()
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> Default for DynamicExecutionContext<Prov, Ptr> {
  fn default() -> Self {
    Self::new()
  }
}

type StratumInputs = HashMap<usize, HashSet<(String, usize)>>;

type StratumOutputs = HashMap<usize, HashSet<String>>;

type RelationLifespan = HashMap<String, usize>;

struct StrataInformation {
  stratum_inputs: StratumInputs,
  stratum_outputs: StratumOutputs,
  relation_lifespan: RelationLifespan,
}

fn supports_result_reuse(program: &ram::Program, runtime: &RuntimeEnvironment) -> bool {
  // Early stopping, implicit entity updates, and side effects cannot be tracked by RAM dependencies.
  runtime.stopping_criteria.get_iter_limit().is_none()
    && !runtime.stopping_criteria.stop_when_goal_relation_non_empty()
    && program.adt_variant_registry.iter().next().is_none()
    && (&runtime.function_registry)
      .into_iter()
      .all(|(name, _)| runtime.function_registry.is_builtin(name))
    && runtime
      .predicate_registry
      .iter()
      .all(|(name, _)| runtime.predicate_registry.is_builtin(name))
    && runtime
      .aggregate_registry
      .iter()
      .all(|(name, _)| runtime.aggregate_registry.is_builtin(name))
    && program
      .strata
      .iter()
      .flat_map(|stratum| &stratum.updates)
      .all(|update| !uses_random_sampling(&update.dataflow))
}

fn uses_random_sampling(flow: &ram::Dataflow) -> bool {
  matches!(flow, ram::Dataflow::Reduce(reduce) if matches!(reduce.aggregator.as_str(), "uniform" | "categorical"))
    || flow.children().any(uses_random_sampling)
}

fn stratum_inputs_outputs(ram: &ram::Program) -> StrataInformation {
  // Cache stratum input output
  let mut stratum_inputs = HashMap::<usize, HashSet<(String, usize)>>::new();
  let mut stratum_outputs = HashMap::<usize, HashSet<String>>::new();

  // Relation start
  let mut relation_lifespan = HashMap::<String, usize>::new();

  // Iterate through the strata
  for (i, stratum) in ram.strata.iter().enumerate() {
    // With dependency, add to input/output
    for dep in stratum.dependency() {
      let dep_stratum = ram.relation_to_stratum[&dep];

      // The `dep_stratum` needs to have an output `dep` relation
      stratum_outputs.entry(dep_stratum).or_default().insert(dep.clone());

      // The `i`-th stratum needs to have `dep` as an input
      stratum_inputs.entry(i).or_default().insert((dep.clone(), dep_stratum));

      // The `i`-th stratum is (currently) the last stratum to use `dep` relation
      relation_lifespan.insert(dep.clone(), i);
    }

    // Regular user requested outputs
    for (predicate, relation) in &stratum.relations {
      if relation.output.is_not_hidden() {
        stratum_outputs.entry(i).or_default().insert(predicate.clone());
      }
    }
  }

  // Generate information
  StrataInformation {
    stratum_inputs,
    stratum_outputs,
    relation_lifespan,
  }
}
