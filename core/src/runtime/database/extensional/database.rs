use std::collections::*;

use itertools::{Either, Itertools};

use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::compiler::ram;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug)]
pub struct ExtensionalDatabase<Prov: Provenance> {
  /// Configuration of whether we perform type check when adding facts
  pub type_check: bool,

  /// Disjunction count; the count used for automatically generating mutual exclusion ID.
  pub disjunction_count: usize,

  /// Types of relations
  pub relation_types: HashMap<String, TupleType>,

  /// Loaded Files
  pub input_file_registry: io::InputFileRegistry,

  /// Extensional relations
  pub extensional_relations: HashMap<String, ExtensionalRelation<Prov>>,

  /// Flag for whether is internalized
  pub internalized: bool,
}

impl<Prov: Provenance> ExtensionalDatabase<Prov> {
  /// Create a new extensional database
  pub fn new() -> Self {
    Self {
      type_check: true,
      disjunction_count: 0,
      relation_types: HashMap::new(),
      input_file_registry: io::InputFileRegistry::new(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn new_with_options(type_check: bool) -> Self {
    Self {
      type_check,
      disjunction_count: 0,
      relation_types: HashMap::new(),
      input_file_registry: io::InputFileRegistry::new(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn clone_with_new_provenance<Prov2: Provenance>(&self) -> ExtensionalDatabase<Prov2>
  where
    Prov2::InputTag: ConvertFromInputTag<Prov::InputTag>,
  {
    ExtensionalDatabase {
      type_check: self.type_check,
      disjunction_count: self.disjunction_count,
      relation_types: self.relation_types.clone(),
      input_file_registry: self.input_file_registry.clone(),
      extensional_relations: self
        .extensional_relations
        .iter()
        .map(|(pred, rel)| {
          let new_rel = rel.clone_with_new_provenance();
          (pred.clone(), new_rel)
        })
        .collect(),
      internalized: false,
    }
  }

  pub fn with_relation_types<I>(types: I) -> Self
  where
    I: Iterator<Item = (String, TupleType)>,
  {
    Self {
      type_check: true,
      disjunction_count: 0,
      relation_types: types.collect(),
      input_file_registry: io::InputFileRegistry::new(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn with_relation_types_and_options<I>(types: I, type_check: bool) -> Self
  where
    I: Iterator<Item = (String, TupleType)>,
  {
    Self {
      type_check,
      disjunction_count: 0,
      relation_types: types.collect(),
      input_file_registry: io::InputFileRegistry::new(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn type_of(&self, relation: &str) -> Option<TupleType> {
    self.relation_types.get(relation).cloned()
  }

  pub fn tap_relation(&mut self, relation: &str) {
    self.extensional_relations.entry(relation.to_string()).or_default();
  }

  pub fn has_relation(&self, relation: &str) -> bool {
    self.extensional_relations.contains_key(relation)
  }

  pub fn add_dynamic_input_facts<T>(
    &mut self,
    relation: &str,
    facts: Vec<(DynamicInputTag, T)>,
  ) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let facts: Vec<(_, Tuple)> = facts.into_iter().map(|(tag, tup)| (tag, tup.into())).collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_dynamic_input_facts(facts);
    Ok(())
  }

  pub fn add_facts<T>(&mut self, relation: &str, facts: Vec<T>) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let facts: Vec<Tuple> = facts.into_iter().map(|tup| tup.into()).collect();
    self.check_tuples_type(relation, facts.iter())?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_facts(facts);
    Ok(())
  }

  pub fn add_static_input_facts(
    &mut self,
    relation: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
  ) -> Result<(), DatabaseError> {
    let facts: Vec<(_, Tuple)> = facts.into_iter().collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    Ok(())
  }

  pub fn get_dynamic_collection(&self, relation: &str) -> Option<DynamicCollectionRef<'_, Prov>> {
    self.extensional_relations.get(relation).map(|r| r.internal.as_ref())
  }

  pub fn pop_dynamic_collection(&mut self, relation: &str) -> Option<DynamicCollection<Prov>> {
    self.extensional_relations.remove(relation).map(|r| r.internal)
  }

  pub fn populate_program_facts(
    &mut self,
    env: &RuntimeEnvironment,
    program: &ram::Program,
  ) -> Result<(), DatabaseError> {
    self.populate_relations(env, program.relations())
  }

  pub fn populate_relations<'a>(
    &mut self,
    env: &RuntimeEnvironment,
    relations: impl IntoIterator<Item = &'a ram::Relation>,
  ) -> Result<(), DatabaseError> {
    let relations = relations.into_iter().collect::<Vec<_>>();
    self
      .input_file_registry
      .load_files(relations.iter().filter_map(|r| r.input_file.as_ref()))
      .map_err(DatabaseError::IO)?;

    // Iterate through all relations declared in the program
    for relation in relations {
      // Create a relation in the EDB if the relation is populating EDB
      if relation.populates_edb() {
        self.create_relation(relation);
      }

      if let Some(edb_relation) = self.extensional_relations.get_mut(&relation.predicate)
        && !edb_relation.set_program_facts(&relation.facts)
      {
        return Err(DatabaseError::NewProgramFacts {
          relation: relation.predicate.clone(),
        });
      }

      // Check if we need to load external facts (from files or databases)
      if let Some(input_file_config) = &relation.input_file {
        let loaded_file_content = self.input_file_registry.get(input_file_config.file_path()).unwrap(); // unwrap since this has to be populated before
        let edb_relation = self.extensional_relations.get_mut(&relation.predicate).unwrap(); // unwrap is okay since the relation is definitely created
        edb_relation.load_from_file(env, input_file_config, loaded_file_content, &relation.tuple_type)?;
      }
    }

    // Return
    Ok(())
  }

  pub fn need_update_relations(&self) -> HashSet<String> {
    if self.internalized {
      self
        .extensional_relations
        .iter()
        .filter(|(_, r)| !r.internalized)
        .map(|(r, _)| r)
        .cloned()
        .collect()
    } else {
      HashSet::new()
    }
  }

  pub fn internalize(&mut self, env: &RuntimeEnvironment, ctx: &Prov) {
    for (_, relation) in self.relations_in_tagging_order(ctx) {
      relation.internalize(env, ctx);
    }
    self.internalized = true
  }

  pub fn internalize_with_monitor<M: Monitor<Prov>>(&mut self, env: &RuntimeEnvironment, ctx: &Prov, m: &M) {
    for (name, relation) in self.relations_in_tagging_order(ctx) {
      m.observe_loading_relation(name);
      m.observe_loading_relation_from_edb(name);
      relation.internalize_with_monitor(env, ctx, m);
    }
    self.internalized = true
  }

  fn relations_in_tagging_order(
    &mut self,
    ctx: &Prov,
  ) -> impl Iterator<Item = (&String, &mut ExtensionalRelation<Prov>)> {
    let relations = self.extensional_relations.iter_mut();
    if ctx.requires_stable_input_order() {
      Either::Left(relations.sorted_unstable_by_key(|(name, _)| *name))
    } else {
      Either::Right(relations)
    }
  }

  fn check_tuples_type<'a, I>(&self, relation: &str, iter: I) -> Result<(), DatabaseError>
  where
    I: Iterator<Item = &'a Tuple>,
  {
    // Check types
    if self.type_check {
      // Get the tuple type
      if let Some(tuple_type) = self.relation_types.get(relation) {
        // Iterate through all tuples
        for tuple in iter {
          if !tuple_type.matches(tuple) {
            return Err(DatabaseError::TypeError {
              relation: relation.to_string(),
              relation_type: tuple_type.clone(),
              tuple: tuple.clone(),
            });
          }
        }
        Ok(())
      } else {
        Err(DatabaseError::UnknownRelation {
          relation: relation.to_string(),
        })
      }
    } else {
      Ok(())
    }
  }

  fn create_relation(&mut self, relation: &ram::Relation) {
    self
      .extensional_relations
      .entry(relation.predicate.clone())
      .or_insert_with(|| ExtensionalRelation::new_with_metadata(relation.storage.clone()));
  }
}

impl<Prov: Provenance> Default for ExtensionalDatabase<Prov> {
  fn default() -> Self {
    Self::new()
  }
}

impl<Prov> ExtensionalDatabase<Prov>
where
  Prov: Provenance<InputTag = InputExclusiveProb>,
{
  pub fn add_exclusive_probabilistic_facts<T>(
    &mut self,
    relation: &str,
    facts: Vec<(f64, T)>,
  ) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let exclusion_id = self.disjunction_count;
    let facts: Vec<(_, Tuple)> = facts
      .into_iter()
      .map(|(prob, tup)| (Some(InputExclusiveProb::new(prob, Some(exclusion_id))), tup.into()))
      .collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    self.disjunction_count += 1;
    Ok(())
  }

  pub fn add_probabilistic_facts<T>(&mut self, relation: &str, facts: Vec<(f64, T)>) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let facts: Vec<(_, Tuple)> = facts
      .into_iter()
      .map(|(prob, tup)| (Some(InputExclusiveProb::new(prob, None)), tup.into()))
      .collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    Ok(())
  }
}
