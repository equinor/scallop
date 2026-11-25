use pyo3::exceptions::*;
use pyo3::{prelude::*, IntoPyObjectExt};

use scallop_core::common::foreign_tensor::*;
use scallop_core::common::tuple::Tuple;
use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value::Value;
use scallop_core::common::value_type::ValueType;
use scallop_core::utils;

use super::error::*;
use super::runtime::*;
use super::tensor::*;

pub fn to_python_tuple(tup: &Tuple, env: &PythonRuntimeEnvironment) -> Option<Py<PyAny>> {
  Python::attach(|py| scallopy_interop::to_python_tuple(py, tup, &|py, val| to_python_extra(py, val, env)).ok())
}

pub fn to_python_value(val: &Value, env: &PythonRuntimeEnvironment) -> Option<Py<PyAny>> {
  Python::attach(|py| scallopy_interop::to_python_value(py, val, &|py, val| to_python_extra(py, val, env)).ok())
}

fn to_python_extra(py: Python<'_>, val: &Value, env: &PythonRuntimeEnvironment) -> PyResult<Py<PyAny>> {
  use Value::*;
  match val {
    Symbol(s) => env.symbol_registry.get_symbol(*s).into_py_any(py),
    SymbolString(s) => s.into_py_any(py),
    DateTime(d) => d.to_string().into_py_any(py),
    Duration(d) => d.to_string().into_py_any(py),
    Entity(e) => e.into_py_any(py),
    EntityString(_) => panic!("[Internal Error] Entity string could not be turned to python value"),
    Tensor(t) => tensor_to_py_object(t.clone()),
    TensorValue(v) => {
      // The registry can call into Python while holding its lock.
      let tensor = py
        .detach(|| env.tensor_registry.eval(v))
        .ok_or_else(|| PyTypeError::new_err("Cannot evaluate tensor"))?;
      tensor_to_py_object(tensor)
    }
    _ => unreachable!("scalar values are converted by scallopy-interop"),
  }
}

pub fn from_python_tuple(v: &Bound<'_, PyAny>, ty: &TupleType, env: &PythonRuntimeEnvironment) -> PyResult<Tuple> {
  scallopy_interop::from_python_tuple(v, ty, &|v, ty| from_python_extra(v, ty, env))
}

pub fn from_python_value(v: &Bound<'_, PyAny>, ty: &ValueType, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  scallopy_interop::from_python_value(v, ty, &|v, ty| from_python_extra(v, ty, env))
}

fn from_python_extra(v: &Bound<'_, PyAny>, ty: &ValueType, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  match ty {
    ValueType::Str => panic!("[Internal Error] Cannot convert python value into static string"),
    ValueType::Symbol => {
      let symbol_str: String = v.extract()?;
      let id = env.symbol_registry.register(symbol_str);
      Ok(Value::Symbol(id))
    }
    ValueType::DateTime => {
      let string = v.extract()?;
      let dt = utils::parse_date_time_string(string)
        .ok_or(PyTypeError::new_err(format!("Cannot parse into DateTime: {}", string)))?;
      Ok(Value::DateTime(dt))
    }
    ValueType::Duration => {
      let string = v.extract()?;
      let dt = utils::parse_duration_string(string)
        .ok_or(PyTypeError::new_err(format!("Cannot parse into Duration: {}", string)))?;
      Ok(Value::Duration(dt))
    }
    ValueType::Entity => {
      let entity_string: String = v.extract()?;
      Ok(Value::EntityString(entity_string))
    }
    ValueType::Tensor => tensor_from_py_object(v, env),
    _ => unreachable!("scalar types are converted by scallopy-interop"),
  }
}

fn tensor_from_py_object(pyobj: &Bound<'_, PyAny>, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  let py_tensor = Tensor::from_py_value(pyobj);
  let symbol = pyobj
    .py()
    .detach(|| env.tensor_registry.register(DynamicExternalTensor::new(py_tensor)))
    .ok_or(BindingError::CannotRegisterTensor)?;
  Ok(Value::TensorValue(symbol.into()))
}

fn tensor_to_py_object(tensor: DynamicExternalTensor) -> PyResult<Py<PyAny>> {
  tensor
    .cast::<Tensor>()
    .map(|t| t.to_py_value())
    .ok_or_else(|| PyTypeError::new_err("Cannot convert external tensor to Python"))
}
