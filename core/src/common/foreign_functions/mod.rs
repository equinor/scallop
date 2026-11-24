//! A library of foreign functions

use super::foreign_function::*;
use super::type_family::*;
use super::value::*;
use super::value_type::*;

mod abs;
mod atan2;
mod ceil;
mod datetime_day;
mod datetime_month;
mod datetime_month0;
mod datetime_year;
mod dim;
mod dot;
mod floor;
mod format;
mod hash;
mod max;
mod min;
mod parse_entity;
mod pow;
mod powf;
mod sign;
mod string_char_at;
mod string_concat;
mod string_index_of;
mod string_length;
mod string_lower;
mod string_replace;
mod string_trim;
mod string_upper;
mod substring;

pub use abs::*;
pub use atan2::*;
pub use ceil::*;
pub use datetime_day::*;
pub use datetime_month::*;
pub use datetime_month0::*;
pub use datetime_year::*;
pub use dim::*;
pub use dot::*;
pub use floor::*;
pub use format::*;
pub use hash::*;
pub use max::*;
pub use min::*;
pub use parse_entity::*;
pub use pow::*;
pub use powf::*;
pub use sign::*;
pub use string_char_at::*;
pub use string_concat::*;
pub use string_index_of::*;
pub use string_length::*;
pub use string_lower::*;
pub use string_replace::*;
pub use string_trim::*;
pub use string_upper::*;
pub use substring::*;

macro_rules! unary_float_functions {
  ($($ty:ident => $name:literal : $method:ident $(if $accepts:expr)?),+ $(,)?) => {
    $(
      #[doc = concat!("`extern fn $", $name, "<T: Float>(x: T) -> T`")]
      #[derive(Clone)]
      pub struct $ty;

      impl UnaryFloatFunction for $ty {
        fn name(&self) -> String {
          $name.to_owned()
        }

        unary_float_functions!(@methods $method $(, $accepts)?);
      }
    )+

    pub(super) fn register_unary_float_functions(registry: &mut ForeignFunctionRegistry) {
      $(registry.register($ty).unwrap();)+
    }
  };
  (@methods $method:ident) => {
    fn execute_f32(&self, arg: f32) -> f32 { arg.$method() }
    fn execute_f64(&self, arg: f64) -> f64 { arg.$method() }
  };
  (@methods $method:ident, $accepts:expr) => {
    fn execute_f32_partial(&self, arg: f32) -> Option<f32> {
      ($accepts)(arg).then(|| arg.$method())
    }
    fn execute_f64_partial(&self, arg: f64) -> Option<f64> {
      ($accepts)(arg).then(|| arg.$method())
    }
  };
}

unary_float_functions! {
  Exp => "exp": exp,
  Exp2 => "exp2": exp2,
  Log => "log": ln if |x| x > 0.0,
  Log2 => "log2": log2 if |x| x > 0.0,
  Sin => "sin": sin,
  Cos => "cos": cos,
  Tan => "tan": tan,
  Asin => "asin": asin if |x| (-1.0..=1.0).contains(&x),
  Acos => "acos": acos if |x| (-1.0..=1.0).contains(&x),
  Atan => "atan": atan,
}
