use pyo3::exceptions::PyIndexError;
use pyo3::types::PyTuple;
use pyo3::{prelude::*, IntoPyObjectExt};
use scallop_core::common::foreign_tensor::FromTensor;
use scallop_core::common::{tuple::Tuple, tuple_type::TupleType, value::Value, value_type::ValueType};
use scallop_core::runtime::provenance::{InputDiffProb, InputExclusiveDiffProb};

pub fn to_py<T: for<'py> IntoPyObject<'py>>(value: T) -> PyResult<Py<PyAny>> {
  Python::attach(|py| value.into_py_any(py))
}

pub fn to_python_tuple(
  py: Python<'_>,
  tuple: &Tuple,
  convert_other: &impl Fn(Python<'_>, &Value) -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
  match tuple {
    Tuple::Tuple(items) => {
      let items = items.iter().map(|item| to_python_tuple(py, item, convert_other));
      let items = items.collect::<PyResult<Vec<_>>>()?;
      Ok(PyTuple::new(py, items)?.into_any().unbind())
    }
    Tuple::Value(value) => to_python_value(py, value, convert_other),
  }
}

pub fn from_python_tuple<'py>(
  value: &Bound<'py, PyAny>,
  ty: &TupleType,
  convert_other: &impl Fn(&Bound<'py, PyAny>, &ValueType) -> PyResult<Value>,
) -> PyResult<Tuple> {
  match ty {
    TupleType::Tuple(types) => {
      let tuple = value.cast::<PyTuple>()?;
      if tuple.len() != types.len() {
        return Err(PyIndexError::new_err("Invalid tuple size"));
      }
      tuple
        .iter()
        .zip(types.iter())
        .map(|(item, ty)| from_python_tuple(&item, ty, convert_other))
        .collect::<PyResult<Box<_>>>()
        .map(Tuple::Tuple)
    }
    TupleType::Value(ty) => from_python_value(value, ty, convert_other).map(Tuple::Value),
  }
}

macro_rules! scalar_conversions {
  ($($variant:ident),+ $(,)?) => {
    pub fn to_python_value(
      py: Python<'_>,
      value: &Value,
      convert_other: &impl Fn(Python<'_>, &Value) -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
      match value {
        $(Value::$variant(value) => value.into_py_any(py),)+
        Value::Str(value) => value.into_py_any(py),
        value => convert_other(py, value),
      }
    }

    pub fn from_python_value<'py>(
      value: &Bound<'py, PyAny>,
      ty: &ValueType,
      convert_other: &impl Fn(&Bound<'py, PyAny>, &ValueType) -> PyResult<Value>,
    ) -> PyResult<Value> {
      match ty {
        $(ValueType::$variant => value.extract().map(Value::$variant),)+
        ty => convert_other(value, ty),
      }
    }
  };
}

scalar_conversions!(I8, I16, I32, I64, I128, ISize, U8, U16, U32, U64, U128, USize, F32, F64, Char, Bool, String);

pub fn diff_probability_tag<'py, T>(
  tag: &Bound<'py, PyAny>,
  extract: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<f64>,
) -> PyResult<Option<InputDiffProb<T>>>
where
  T: FromTensor + for<'a> From<&'a Bound<'py, PyAny>>,
{
  Ok(Some(InputDiffProb(extract(tag)?, Some(tag.into()))))
}

pub fn diff_exclusive_probability_tag<'py, T>(
  tag: &Bound<'py, PyAny>,
  extract: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<f64>,
) -> PyResult<Option<InputExclusiveDiffProb<T>>>
where
  T: FromTensor + for<'a> From<&'a Bound<'py, PyAny>>,
{
  let (probability, exclusion): (Bound<'py, PyAny>, Option<usize>) = tag.extract()?;
  if probability.is_none() {
    Ok(None)
  } else {
    Ok(Some(InputExclusiveDiffProb::new(
      extract(&probability)?,
      (&probability).into(),
      exclusion,
    )))
  }
}
