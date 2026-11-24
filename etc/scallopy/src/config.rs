use pyo3::prelude::*;

#[pyfunction]
pub fn torch_tensor_enabled() -> bool {
  cfg!(feature = "torch-tensor")
}
