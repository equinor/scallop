use itertools::Itertools;

use crate::common::tuple::*;
use crate::common::type_family::*;
use crate::common::value::Value;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct SumProdAggregate {
  is_sum: bool,
}

impl SumProdAggregate {
  pub fn sum() -> Self {
    Self { is_sum: true }
  }

  pub fn prod() -> Self {
    Self { is_sum: false }
  }
}

impl From<SumProdAggregate> for DynamicAggregate {
  fn from(val: SumProdAggregate) -> Self {
    DynamicAggregate::SumProd(val)
  }
}

impl Aggregate for SumProdAggregate {
  type Aggregator<P: Provenance> = SumProdAggregator;

  fn name(&self) -> String {
    if self.is_sum {
      "sum".to_string()
    } else {
      "prod".to_string()
    }
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![
        ("A".to_string(), GenericTypeFamily::possibly_empty_tuple()),
        ("T".to_string(), GenericTypeFamily::type_family(TypeFamily::Number)),
      ]
      .into_iter()
      .collect(),
      arg_type: BindingTypes::generic("A"),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::generic("T"),
      allow_exclamation_mark: true,
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    SumProdAggregator {
      is_sum: self.is_sum,
      non_multi_world: info.has_exclamation_mark,
      num_args: info.arg_var_types.len(),
      value_type: info.input_var_types[0].clone(),
    }
  }
}

#[derive(Clone)]
pub struct SumProdAggregator {
  is_sum: bool,
  non_multi_world: bool,
  num_args: usize,
  value_type: ValueType,
}

impl SumProdAggregator {
  pub(crate) fn numeric_value<'a>(&self, tuple: &'a Tuple) -> &'a Tuple {
    if self.num_args > 0 {
      &tuple[self.num_args]
    } else {
      tuple
    }
  }

  pub(crate) fn combine_numeric(&self, accumulator: &Tuple, input: &Tuple) -> Tuple {
    use Value::*;
    macro_rules! combine_integer {
      ($a:expr, $b:expr) => {
        if self.is_sum {
          $a.wrapping_add(*$b)
        } else {
          $a.wrapping_mul(*$b)
        }
      };
    }
    macro_rules! combine_float {
      ($a:expr, $b:expr) => {
        if self.is_sum {
          $a + $b
        } else {
          $a * $b
        }
      };
    }
    let result = match (accumulator, input) {
      (Tuple::Value(a), Tuple::Value(b)) => match (a, b) {
        (I8(a), I8(b)) => I8(combine_integer!(a, b)),
        (I16(a), I16(b)) => I16(combine_integer!(a, b)),
        (I32(a), I32(b)) => I32(combine_integer!(a, b)),
        (I64(a), I64(b)) => I64(combine_integer!(a, b)),
        (I128(a), I128(b)) => I128(combine_integer!(a, b)),
        (ISize(a), ISize(b)) => ISize(combine_integer!(a, b)),
        (U8(a), U8(b)) => U8(combine_integer!(a, b)),
        (U16(a), U16(b)) => U16(combine_integer!(a, b)),
        (U32(a), U32(b)) => U32(combine_integer!(a, b)),
        (U64(a), U64(b)) => U64(combine_integer!(a, b)),
        (U128(a), U128(b)) => U128(combine_integer!(a, b)),
        (USize(a), USize(b)) => USize(combine_integer!(a, b)),
        (F32(a), F32(b)) => F32(combine_float!(a, b)),
        (F64(a), F64(b)) => F64(combine_float!(a, b)),
        _ => unreachable!("the numeric aggregate inputs share a scalar numeric type"),
      },
      _ => unreachable!("the numeric aggregate value must be scalar"),
    };
    Tuple::Value(result)
  }

  pub fn sum<T>(non_multi_world: bool) -> Self
  where
    ValueType: FromType<T>,
  {
    Self {
      is_sum: true,
      non_multi_world,
      num_args: 0,
      value_type: ValueType::from_type(),
    }
  }

  pub fn prod<T>(non_multi_world: bool) -> Self
  where
    ValueType: FromType<T>,
  {
    Self {
      is_sum: false,
      non_multi_world,
      num_args: 0,
      value_type: ValueType::from_type(),
    }
  }
}

impl SumProdAggregator {
  pub fn perform_sum_prod<'a, I: Iterator<Item = &'a Tuple>>(&self, i: I) -> Tuple {
    if self.num_args > 0 {
      let iterator = i.map(|t| &t[self.num_args]);
      if self.is_sum {
        self.value_type.sum(iterator)
      } else {
        self.value_type.prod(iterator)
      }
    } else {
      if self.is_sum {
        self.value_type.sum(i)
      } else {
        self.value_type.prod(i)
      }
    }
  }
}

impl<P: Provenance> Aggregator<P> for SumProdAggregator {
  default fn aggregate_implicit_group(
    &self,
    prov: &P,
    env: &RuntimeEnvironment,
    batch: DynamicElements<P>,
  ) -> DynamicElements<P> {
    self.aggregate(prov, env, batch)
  }

  default fn aggregate(&self, prov: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      let res = self.perform_sum_prod(batch.iter_tuples());
      vec![DynamicElement::new(res, prov.one())]
    } else {
      let mut result = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let res = self.perform_sum_prod(chosen_set.iter().map(|i| &batch[*i].tuple));
        if let Some(tag) = super::chosen_set_tag(prov, &batch, &chosen_set) {
          result.push(DynamicElement::new(res, tag));
        }
      }
      result
    }
  }
}
