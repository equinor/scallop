use std::sync::Arc;

use pyo3::{Py, PyAny, Python};
use scallop_core::runtime::provenance;

/// The custom tag which holds an arbitrary python object.
#[derive(Clone, Debug)]
pub struct CustomTag(pub Arc<Py<PyAny>>);

impl CustomTag {
  /// Create a new custom tag with a python object.
  pub fn new(tag: Py<PyAny>) -> Self {
    Self(tag.into())
  }
}

impl std::fmt::Display for CustomTag {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self, f)
  }
}

/// Custom tag is a tag
impl provenance::Tag for CustomTag {}

/// The custom provenance which is a wrapper of a python class
#[derive(Clone, Debug)]
pub struct CustomProvenance(pub Arc<Py<PyAny>>);

impl provenance::Provenance for CustomProvenance {
  type Tag = CustomTag;

  type InputTag = CustomTag;

  type OutputTag = CustomTag;

  fn name(&self) -> String {
    Python::attach(|py| self.0.call_method(py, "name", (), None).unwrap().to_string())
  }

  /// Invoking the provenance's tagging function on the input tag
  fn tagging_fn(&self, i: Self::InputTag) -> Self::Tag {
    Python::attach(|py| Self::Tag::new(self.0.call_method(py, "tagging_fn", (i.0.bind(py),), None).unwrap()))
  }

  /// Invoking the provenance's recover function on an internal tag
  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    Python::attach(|py| {
      self
        .0
        .call_method(py, "recover_fn", (t.0.bind(py),), None)
        .map(CustomTag::new)
        .unwrap()
    })
  }

  /// Invoking the provenance's discard function on an internal tag
  fn discard(&self, t: &Self::Tag) -> bool {
    Python::attach(|py| {
      self
        .0
        .call_method(py, "discard", (t.0.bind(py),), None)
        .unwrap()
        .extract(py)
        .unwrap()
    })
  }

  fn zero(&self) -> Self::Tag {
    Python::attach(|py| {
      Self::Tag::new(
        self
          .0
          .call_method(py, "zero", (), None)
          .expect("Python error in `zero`")
          .extract(py)
          .expect("Python error in `zero`"),
      )
    })
  }

  fn one(&self) -> Self::Tag {
    Python::attach(|py| {
      Self::Tag::new(
        self
          .0
          .call_method(py, "one", (), None)
          .expect("Python error in `one`")
          .extract(py)
          .expect("Python error in `one`"),
      )
    })
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    Python::attach(|py| {
      let input = (t1.0.bind(py), t2.0.bind(py));
      Self::Tag::new(
        self
          .0
          .call_method(py, "add", input, None)
          .expect("Python error in `add`")
          .extract(py)
          .expect("Python error in `add`"),
      )
    })
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    Python::attach(|py| {
      let input = (t1.0.bind(py), t2.0.bind(py));
      Self::Tag::new(
        self
          .0
          .call_method(py, "mult", input, None)
          .expect("Python error in `mult`")
          .extract(py)
          .expect("Python error in `mult`"),
      )
    })
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Python::attach(|py| {
      let input = (t.0.bind(py),);
      Some(Self::Tag::new(
        self
          .0
          .call_method(py, "negate", input, None)
          .expect("Python error in `negate`")
          .extract(py)
          .expect("Python error in `negate`"),
      ))
    })
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    Python::attach(|py| {
      let input = (t_old.0.bind(py), t_new.0.bind(py));
      self
        .0
        .call_method(py, "saturated", input, None)
        .unwrap()
        .extract(py)
        .unwrap()
    })
  }
}
