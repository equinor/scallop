use std::sync::LazyLock;

use crate::common::value_type::*;

pub static ADD_TYPING_RULES: LazyLock<Vec<(ValueType, ValueType, ValueType)>> = LazyLock::new(|| {
  use ValueType::*;
  vec![
    (I32, I32, I32), // Prioritize
    (I8, I8, I8),
    (I16, I16, I16),
    (I64, I64, I64),
    (I128, I128, I128),
    (ISize, ISize, ISize),
    (U32, U32, U32), // Prioritize
    (U8, U8, U8),
    (U16, U16, U16),
    (U64, U64, U64),
    (U128, U128, U128),
    (USize, USize, USize),
    (F32, F32, F32), // Prioritize
    (F64, F64, F64),
    (String, String, String),
    (DateTime, Duration, DateTime),
    (Duration, DateTime, DateTime),
    (Duration, Duration, Duration),
    (Tensor, Tensor, Tensor),
    (Tensor, F64, Tensor),
    (F64, Tensor, Tensor),
  ]
});

pub static SUB_TYPING_RULES: LazyLock<Vec<(ValueType, ValueType, ValueType)>> = LazyLock::new(|| {
  use ValueType::*;
  vec![
    (I32, I32, I32), // Prioritize
    (I8, I8, I8),
    (I16, I16, I16),
    (I64, I64, I64),
    (I128, I128, I128),
    (ISize, ISize, ISize),
    (U32, U32, U32), // Prioritize
    (U8, U8, U8),
    (U16, U16, U16),
    (U64, U64, U64),
    (U128, U128, U128),
    (USize, USize, USize),
    (F32, F32, F32), // Prioritize
    (F64, F64, F64),
    (DateTime, Duration, DateTime),
    (DateTime, DateTime, Duration),
    (Duration, Duration, Duration),
    (Tensor, Tensor, Tensor),
    (Tensor, F64, Tensor),
    (F64, Tensor, Tensor),
  ]
});

pub static MULT_TYPING_RULES: LazyLock<Vec<(ValueType, ValueType, ValueType)>> = LazyLock::new(|| {
  use ValueType::*;
  vec![
    (I32, I32, I32), // Prioritize
    (I8, I8, I8),
    (I16, I16, I16),
    (I64, I64, I64),
    (I128, I128, I128),
    (ISize, ISize, ISize),
    (U32, U32, U32), // Prioritize
    (U8, U8, U8),
    (U16, U16, U16),
    (U64, U64, U64),
    (U128, U128, U128),
    (USize, USize, USize),
    (F32, F32, F32), // Prioritize
    (F64, F64, F64),
    (Duration, I32, Duration),
    (I32, Duration, Duration),
    (Tensor, Tensor, Tensor),
    (Tensor, F64, Tensor),
    (F64, Tensor, Tensor),
  ]
});

pub static DIV_TYPING_RULES: LazyLock<Vec<(ValueType, ValueType, ValueType)>> = LazyLock::new(|| {
  use ValueType::*;
  vec![
    (I32, I32, I32), // Prioritize
    (I8, I8, I8),
    (I16, I16, I16),
    (I64, I64, I64),
    (I128, I128, I128),
    (ISize, ISize, ISize),
    (U32, U32, U32), // Prioritize
    (U8, U8, U8),
    (U16, U16, U16),
    (U64, U64, U64),
    (U128, U128, U128),
    (USize, USize, USize),
    (F32, F32, F32), // Prioritize
    (F64, F64, F64),
    (Duration, I32, Duration),
  ]
});

pub static MOD_TYPING_RULES: LazyLock<Vec<(ValueType, ValueType, ValueType)>> = LazyLock::new(|| {
  use ValueType::*;
  vec![
    (I32, I32, I32), // Prioritize
    (I8, I8, I8),
    (I16, I16, I16),
    (I64, I64, I64),
    (I128, I128, I128),
    (ISize, ISize, ISize),
    (U32, U32, U32), // Prioritize
    (U8, U8, U8),
    (U16, U16, U16),
    (U64, U64, U64),
    (U128, U128, U128),
    (USize, USize, USize),
    (F32, F32, F32), // Prioritize
    (F64, F64, F64),
  ]
});

pub static COMPARE_TYPING_RULES: LazyLock<Vec<(ValueType, ValueType)>> = LazyLock::new(|| {
  use ValueType::*;
  vec![
    (I32, I32), // Prioritize
    (I8, I8),
    (I16, I16),
    (I64, I64),
    (I128, I128),
    (ISize, ISize),
    (U32, U32), // Prioritize
    (U8, U8),
    (U16, U16),
    (U64, U64),
    (U128, U128),
    (USize, USize),
    (F32, F32), // Prioritize
    (F64, F64),
    (Duration, Duration),
    (DateTime, DateTime),
  ]
});
