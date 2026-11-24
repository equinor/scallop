use std::sync::Arc;

use pyo3::{prelude::*, IntoPyObjectExt};

use scallop_core::common::foreign_tensor::*;

use super::tensor::*;

#[derive(Clone)]
pub struct ExtTag {
  tag: Arc<Py<PyAny>>,
}

impl FromTensor for ExtTag {
  fn from_tensor(tensor: DynamicExternalTensor) -> Option<Self> {
    tensor.cast::<Tensor>().map(|t| t.to_py_value().into())
  }
}

impl From<Py<PyAny>> for ExtTag {
  fn from(tag: Py<PyAny>) -> Self {
    Self { tag: tag.into() }
  }
}

impl From<&Bound<'_, PyAny>> for ExtTag {
  fn from(tag: &Bound<'_, PyAny>) -> Self {
    tag.clone().unbind().into()
  }
}

impl From<ExtTag> for Py<PyAny> {
  fn from(val: ExtTag) -> Self {
    Arc::try_unwrap(val.tag).unwrap_or_else(|tag| Python::attach(|py| tag.clone_ref(py)))
  }
}

pub trait ExtTagVec {
  fn into_vec(self) -> Vec<Py<PyAny>>;

  fn into_none_prepended_vec(self) -> Vec<Py<PyAny>>;
}

impl ExtTagVec for Vec<ExtTag> {
  fn into_vec(self) -> Vec<Py<PyAny>> {
    self.into_iter().map(Into::into).collect()
  }

  fn into_none_prepended_vec(self) -> Vec<Py<PyAny>> {
    let none: Option<Py<PyAny>> = None;
    std::iter::once(Python::attach(|py| none.into_py_any(py).unwrap()))
      .chain(self.into_iter().map(Into::into))
      .collect()
  }
}

pub trait ExtTagOption {
  fn into_option(self) -> Option<Py<PyAny>>;
}

impl ExtTagOption for Option<ExtTag> {
  fn into_option(self) -> Option<Py<PyAny>> {
    self.map(Into::into)
  }
}

impl ExtTagOption for Option<&ExtTag> {
  fn into_option(self) -> Option<Py<PyAny>> {
    self.map(|v| Python::attach(|py| v.tag.clone_ref(py)))
  }
}
