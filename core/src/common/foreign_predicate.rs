use std::collections::*;

use dyn_clone::*;

use crate::compiler::front::analyzers::tagged_rule::TagPredicate;
use crate::runtime::env::RuntimeEnvironment;

use super::foreign_predicates as fps;
use super::input_tag::*;
use super::value::*;
use super::value_type::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Binding {
  Free,
  Bound,
}

impl std::fmt::Display for Binding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Free => f.write_str("f"),
      Self::Bound => f.write_str("b"),
    }
  }
}

impl Binding {
  pub fn is_bound(&self) -> bool {
    matches!(self, Self::Bound)
  }

  pub fn is_free(&self) -> bool {
    matches!(self, Self::Free)
  }
}

// /// The identifier of a foreign predicate in a registry
// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
// pub struct ForeignPredicateIdentifier {
//   identifier: String,
//   types: Box<[ValueType]>,
//   binding_pattern: BindingPattern,
// }

// impl std::fmt::Display for ForeignPredicateIdentifier {
//   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//     f.write_fmt(format_args!(
//       "pred {}[{}]({})",
//       self.identifier,
//       self.binding_pattern,
//       self
//         .types
//         .iter()
//         .map(|t| format!("{}", t))
//         .collect::<Vec<_>>()
//         .join(", ")
//     ))
//   }
// }

/// A binding pattern for a predicate, e.g. bbf
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct BindingPattern {
  pattern: Box<[Binding]>,
}

impl std::fmt::Display for BindingPattern {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for binding in &*self.pattern {
      binding.fmt(f)?;
    }
    Ok(())
  }
}

impl BindingPattern {
  /// Create a new binding pattern given the predicate arity and number of bounded variables
  pub fn new(arity: usize, num_bounded: usize) -> Self {
    assert!(num_bounded <= arity);
    Self {
      pattern: (0..arity)
        .map(|i| if i < num_bounded { Binding::Bound } else { Binding::Free })
        .collect(),
    }
  }

  /// Get the length of the binding pattern
  pub fn len(&self) -> usize {
    self.pattern.len()
  }

  pub fn is_empty(&self) -> bool {
    self.pattern.is_empty()
  }

  /// Check if all argument needs to be bounded
  pub fn is_bounded(&self) -> bool {
    self.pattern.iter().all(|p| p.is_bound())
  }

  /// Check if all arguments are free
  pub fn is_free(&self) -> bool {
    self.pattern.iter().all(|p| p.is_free())
  }

  pub fn iter(&self) -> std::slice::Iter<'_, Binding> {
    self.pattern.iter()
  }
}

impl std::ops::Index<usize> for BindingPattern {
  type Output = Binding;

  fn index(&self, index: usize) -> &Self::Output {
    &self.pattern[index]
  }
}

/// The foreign predicate for a runtime implementation
///
/// The arguments of a foreign predicate can be marked as "need to be bounded"
/// or free.
/// We assume the bounded arguments are always placed before the free arguments.
/// During runtime, we expect the foreign predicate to take in all bounded
/// variables as input, and produce the free variables, along with a tag associated
/// with the tuple.
pub trait ForeignPredicate: DynClone {
  /// The name of the predicate
  fn name(&self) -> String;

  /// Generic type parameters
  fn generic_type_parameters(&self) -> Vec<ValueType> {
    vec![]
  }

  fn internal_name(&self) -> String {
    let name = self.name();
    let type_params = self.generic_type_parameters();
    if !type_params.is_empty() {
      format!(
        "{}#{}",
        name,
        type_params
          .into_iter()
          .map(|t| t.to_string())
          .collect::<Vec<_>>()
          .join("#")
      )
    } else {
      name.to_string()
    }
  }

  /// The arity of the predicate (i.e. number of arguments)
  fn arity(&self) -> usize;

  /// The type of the `i`-th argument
  ///
  /// Should panic if `i` is larger than or equal to the arity
  fn argument_type(&self, i: usize) -> ValueType;

  /// The number of bounded arguments
  fn num_bounded(&self) -> usize;

  /// The number of free arguments
  fn num_free(&self) -> usize {
    self.arity() - self.num_bounded()
  }

  /// Get a vector of the argument types
  fn argument_types(&self) -> Vec<ValueType> {
    (0..self.arity()).map(|i| self.argument_type(i)).collect()
  }

  /// Get a vector of free argument types
  fn free_argument_types(&self) -> Vec<ValueType> {
    (self.num_bounded()..self.arity())
      .map(|i| self.argument_type(i))
      .collect()
  }

  /// Get an identifier for this predicate
  fn binding_pattern(&self) -> BindingPattern {
    BindingPattern::new(self.arity(), self.num_bounded())
  }

  /// Evaluate the foreign predicate given a tuple containing bounded variables
  ///
  /// The `bounded` tuple (`Vec<Value>`) should have arity (length) `self.num_bounded()`.
  /// The function returns a sequence of (dynamically) tagged-tuples where the arity is `self.num_free()`
  #[allow(unused_variables)]
  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    panic!(
      "[Internal Error] Missing evaluate function in the foreign predicate `{}`",
      self.name()
    )
  }

  /// Evaluate the foreign predicate given a tuple containing bounded variables and an environment
  #[allow(unused_variables)]
  fn evaluate_with_env(&self, env: &RuntimeEnvironment, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    self.evaluate(bounded)
  }
}

dyn_clone::clone_trait_object!(ForeignPredicate);

/// The dynamic foreign predicate
#[derive(Clone)]
pub struct DynamicForeignPredicate {
  fp: Box<dyn ForeignPredicate + Send + Sync>,
  origin: PredicateOrigin,
}

#[derive(Clone)]
enum PredicateOrigin {
  Custom,
  Builtin,
  RuleTag(TagPredicate),
}

impl DynamicForeignPredicate {
  pub fn new<P: ForeignPredicate + Send + Sync + 'static>(fp: P) -> Self {
    Self {
      fp: Box::new(fp),
      origin: PredicateOrigin::Custom,
    }
  }
}

impl ForeignPredicate for DynamicForeignPredicate {
  fn name(&self) -> String {
    self.fp.name()
  }

  fn generic_type_parameters(&self) -> Vec<ValueType> {
    self.fp.generic_type_parameters()
  }

  fn arity(&self) -> usize {
    self.fp.arity()
  }

  fn argument_type(&self, i: usize) -> ValueType {
    self.fp.argument_type(i)
  }

  fn num_bounded(&self) -> usize {
    self.fp.num_bounded()
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    self.fp.evaluate(bounded)
  }

  fn evaluate_with_env(&self, env: &RuntimeEnvironment, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    self.fp.evaluate_with_env(env, bounded)
  }
}

impl std::fmt::Debug for DynamicForeignPredicate {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ForeignPredicate")
      .field("name", &self.name())
      .field(
        "types",
        &(0..self.arity()).map(|i| self.argument_type(i)).collect::<Vec<_>>(),
      )
      .field("num_bounded", &self.num_bounded())
      .finish()
  }
}

impl std::fmt::Display for DynamicForeignPredicate {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("pred {}(", self.name()))?;
    for i in 0..self.arity() {
      if i > 0 {
        f.write_str(", ")?;
      }
      self.argument_type(i).fmt(f)?;
      if i < self.num_bounded() {
        f.write_str(" [b]")?;
      } else {
        f.write_str(" [f]")?;
      }
    }
    f.write_str(")")
  }
}

/// A foreign predicate registry
#[derive(Clone, Debug)]
pub struct ForeignPredicateRegistry {
  registry: HashMap<String, DynamicForeignPredicate>,
}

impl ForeignPredicateRegistry {
  /// Create an empty foreign predicate registry
  pub fn new() -> Self {
    Self {
      registry: HashMap::new(),
    }
  }

  /// Create a Standard Library foreign predicate registry
  pub fn std() -> Self {
    let mut reg = Self::new();

    // Register all predicates

    // Range
    for value_type in ValueType::integers() {
      reg.register(fps::RangeBBF::new(value_type.clone())).unwrap();
    }

    // Soft comparison operators
    for value_type in ValueType::floats() {
      reg.register(fps::FloatEq::new(value_type.clone())).unwrap();
      reg.register(fps::SoftNumberEq::new(value_type.clone())).unwrap();
      reg.register(fps::SoftNumberNeq::new(value_type.clone())).unwrap();
      reg.register(fps::SoftNumberGt::new(value_type.clone())).unwrap();
      reg.register(fps::SoftNumberLt::new(value_type.clone())).unwrap();
    }

    // String operations
    reg.register(fps::StringCharsBFF::new()).unwrap();
    reg.register(fps::StringIndexBF::new()).unwrap();
    reg.register(fps::StringFindBBFF::new()).unwrap();
    reg.register(fps::StringSplitBBF::new()).unwrap();

    // DateTime
    reg.register(fps::DateTimeYMD::new()).unwrap();

    // Tensor
    reg.register(fps::TensorShape::new()).unwrap();

    // Provenance
    reg.register(fps::NewTagVariable::new()).unwrap();

    for predicate in reg.registry.values_mut() {
      predicate.origin = PredicateOrigin::Builtin;
    }
    reg
  }

  pub(crate) fn is_builtin(&self, name: &str) -> bool {
    self
      .get(name)
      .is_some_and(|p| !matches!(p.origin, PredicateOrigin::Custom))
  }

  pub(crate) fn rule_tags(&self) -> impl Iterator<Item = &TagPredicate> {
    self.registry.values().filter_map(|p| match &p.origin {
      PredicateOrigin::RuleTag(tag) => Some(tag),
      _ => None,
    })
  }

  pub(crate) fn register_rule_tag(&mut self, tag: TagPredicate) -> Result<(), ForeignPredicateError> {
    let id = tag.internal_name();
    let predicate = DynamicForeignPredicate {
      fp: Box::new(tag.clone()),
      origin: PredicateOrigin::RuleTag(tag),
    };
    self.insert(id, predicate)
  }

  /// Register a new foreign predicate in the registry
  pub fn register<P: ForeignPredicate + Send + Sync + 'static>(&mut self, p: P) -> Result<(), ForeignPredicateError> {
    self.insert(p.internal_name(), DynamicForeignPredicate::new(p))
  }

  fn insert(&mut self, id: String, p: DynamicForeignPredicate) -> Result<(), ForeignPredicateError> {
    if self.contains(&id) {
      Err(ForeignPredicateError::AlreadyExisted { id })
    } else {
      self.registry.insert(id, p);
      Ok(())
    }
  }

  /// Check if the registry contains a foreign predicate by using its identifier
  pub fn contains(&self, id: &str) -> bool {
    self.registry.contains_key(id)
  }

  /// Get the foreign predicate
  pub fn get(&self, id: &str) -> Option<&DynamicForeignPredicate> {
    self.registry.get(id)
  }

  pub fn iter<'a>(&'a self) -> hash_map::Iter<'a, String, DynamicForeignPredicate> {
    self.into_iter()
  }
}

impl Default for ForeignPredicateRegistry {
  fn default() -> Self {
    Self::new()
  }
}

impl<'a> IntoIterator for &'a ForeignPredicateRegistry {
  type IntoIter = hash_map::Iter<'a, String, DynamicForeignPredicate>;

  type Item = (&'a String, &'a DynamicForeignPredicate);

  fn into_iter(self) -> Self::IntoIter {
    self.registry.iter()
  }
}

/// THe errors happening when handling foreign predicates
#[derive(Clone, Debug, thiserror::Error)]
pub enum ForeignPredicateError {
  #[error("Foreign predicate `{id}` already existed")]
  AlreadyExisted { id: String },
}
