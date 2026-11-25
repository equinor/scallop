mod constraint;
mod ground;
mod join;

pub use constraint::*;
pub use ground::*;
pub use join::*;

use crate::common::foreign_predicate::{DynamicForeignPredicate, ForeignPredicate};
use crate::common::tuple::Tuple;
use crate::common::value::Value;

use super::*;

fn evaluate_output<Prov: Provenance>(
  predicate: &DynamicForeignPredicate,
  args: &[Value],
  runtime: &RuntimeEnvironment,
  ctx: &Prov,
) -> DynamicElements<Prov> {
  predicate
    .evaluate_with_env(runtime, args)
    .into_iter()
    .filter_map(|(input_tag, values)| {
      let input_tag = Prov::InputTag::from_dynamic_input_tag(&input_tag);
      let tag = ctx.tagging_optional_fn(input_tag);
      let fields = values
        .into_iter()
        .map(|value| runtime.internalize_owned_value(value).map(Tuple::Value))
        .collect::<Option<_>>()?;
      Some(DynamicElement::new(Tuple::Tuple(fields), tag))
    })
    .collect()
}
