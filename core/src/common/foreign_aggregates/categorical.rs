use std::collections::*;

use rand::distr::weighted::WeightedIndex;

use crate::common::value_type::*;
use crate::runtime::env::*;

use super::*;

#[derive(Clone, Default)]
pub struct CategoricalAggregate;

impl CategoricalAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl From<CategoricalAggregate> for DynamicAggregate {
  fn from(val: CategoricalAggregate) -> Self {
    DynamicAggregate::Sampler(DynamicSampleAggregate::new(val))
  }
}

impl SampleAggregate for CategoricalAggregate {
  fn name(&self) -> String {
    "categorical".to_string()
  }

  fn param_types(&self) -> Vec<ParamType> {
    vec![ParamType::Mandatory(ValueType::USize)]
  }

  fn instantiate(&self, info: AggregateInfo) -> DynamicSampler {
    CategoricalSampler {
      k: info.pos_params[0].as_usize(),
    }
    .into()
  }
}

#[derive(Clone)]
pub struct CategoricalSampler {
  k: usize,
}

impl From<CategoricalSampler> for DynamicSampler {
  fn from(val: CategoricalSampler) -> Self {
    DynamicSampler::new(val)
  }
}

impl Sampler for CategoricalSampler {
  fn sampler_type(&self) -> SamplerType {
    SamplerType::WeightOnly
  }

  fn sample_weight_only(&self, rt: &RuntimeEnvironment, weights: Vec<f64>) -> Vec<usize> {
    if weights.len() <= self.k {
      (0..weights.len()).collect()
    } else {
      let dist = WeightedIndex::new(&weights).unwrap();
      (0..self.k)
        .map(|_| rt.random.sample_from(&dist))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
    }
  }
}
