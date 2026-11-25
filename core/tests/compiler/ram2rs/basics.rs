use scallop_core::common::foreign_function::{ForeignFunction, ForeignFunctionParameterType};
use scallop_core::common::value::Value;
use scallop_core::common::value_type::ValueType;
use scallop_core::compiler::front::StringSource;
use scallop_core::compiler::ram::RustEmissionError;
use scallop_core::compiler::{compile_source_to_ram, CompileOptions};

#[test]
fn ram2rs_edge_path() {
  let opt = CompileOptions::default();
  let program = r#"
    rel edge = {(0, 1), (1, 2), (2, 3)}
    rel path(a, b) = edge(a, b) or (path(a, c) and edge(c, b))
  "#;
  let source = StringSource::new(program.into());
  let ram = compile_source_to_ram(source).unwrap();
  ram.to_rs_module(&opt).unwrap();
}

#[test]
fn ram2rs_rejects_custom_functions_even_after_constant_folding() {
  #[derive(Clone)]
  struct Callback;
  impl ForeignFunction for Callback {
    fn name(&self) -> String {
      "callback".into()
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::I32)
    }

    fn execute(&self, _: Vec<Value>) -> Option<Value> {
      Some(Value::I32(42))
    }
  }

  let opt = CompileOptions::default();
  let mut front = scallop_core::compiler::front::FrontContext::new();
  front.register_foreign_function(Callback).unwrap();
  front
    .compile_source(StringSource::new("rel answer(x) = x == $callback()".into()))
    .unwrap();
  let mut back = front.to_back_program();
  back.apply_optimizations(&opt).unwrap();
  let ram = back.to_ram_program(&opt).unwrap();
  assert!(matches!(
    ram.to_rs_module(&opt),
    Err(RustEmissionError::ForeignExtension { kind: "function", name }) if name == "callback"
  ));
}

#[test]
fn ram2rs_rejects_relations_that_need_a_tensor_provider() {
  let ram = compile_source_to_ram(StringSource::new("type samples(Tensor) query samples".into())).unwrap();
  assert!(matches!(
    ram.to_rs_module(&CompileOptions::default()),
    Err(RustEmissionError::TensorRelation(name)) if name == "samples"
  ));
}
