use crate::common::tuple::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Default)]
pub struct EnumerateAggregate;

impl EnumerateAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl From<EnumerateAggregate> for DynamicAggregate {
  fn from(val: EnumerateAggregate) -> Self {
    DynamicAggregate::Enumerate(val)
  }
}

impl Aggregate for EnumerateAggregate {
  type Aggregator<P: Provenance> = EnumerateAggregator;

  fn name(&self) -> String {
    "enumerate".to_string()
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![("T".to_string(), GenericTypeFamily::non_empty_tuple())]
        .into_iter()
        .collect(),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::tuple(vec![
        BindingType::value_type(ValueType::USize),
        BindingType::generic("T"),
      ]),
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, _: AggregateInfo) -> Self::Aggregator<P> {
    EnumerateAggregator
  }
}

#[derive(Clone)]
pub struct EnumerateAggregator;

impl<Prov: Provenance> Aggregator<Prov> for EnumerateAggregator {
  default fn aggregate(&self, _: &Prov, _: &RuntimeEnvironment, batch: DynamicElements<Prov>) -> DynamicElements<Prov> {
    batch
      .into_iter()
      .enumerate()
      .map(|(i, elem)| {
        let tup = Tuple::from_values(std::iter::once(i.into()).chain(elem.tuple.to_values()));
        DynamicElement::new(tup, elem.tag)
      })
      .collect()
  }
}
