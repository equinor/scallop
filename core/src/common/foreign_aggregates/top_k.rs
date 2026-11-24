use crate::common::value_type::*;
use crate::runtime::env::*;

pub use crate::runtime::provenance::{aggregate_top_k_helper, unweighted_aggregate_top_k_helper};

use super::*;

#[derive(Clone)]
pub struct TopKSamplerAggregate {
  is_unique: bool,
}

impl TopKSamplerAggregate {
  pub fn top() -> Self {
    Self { is_unique: false }
  }

  pub fn unique() -> Self {
    Self { is_unique: true }
  }
}

impl From<TopKSamplerAggregate> for DynamicAggregate {
  fn from(val: TopKSamplerAggregate) -> Self {
    DynamicAggregate::Sampler(DynamicSampleAggregate::new(val))
  }
}

impl SampleAggregate for TopKSamplerAggregate {
  fn name(&self) -> String {
    if self.is_unique {
      "unique".to_string()
    } else {
      "top".to_string()
    }
  }

  fn param_types(&self) -> Vec<ParamType> {
    if self.is_unique {
      vec![]
    } else {
      vec![ParamType::Optional(ValueType::USize)]
    }
  }

  fn instantiate(&self, info: AggregateInfo) -> DynamicSampler {
    if self.is_unique {
      TopKSampler { k: 1 }.into()
    } else {
      TopKSampler {
        k: info.pos_params.first().map(|v| v.as_usize()).unwrap_or(1),
      }
      .into()
    }
  }
}

#[derive(Clone)]
pub struct TopKSampler {
  k: usize,
}

impl TopKSampler {
  pub fn new(k: usize) -> Self {
    Self { k }
  }
}

impl From<TopKSampler> for DynamicSampler {
  fn from(val: TopKSampler) -> Self {
    DynamicSampler::new(val)
  }
}

impl Sampler for TopKSampler {
  fn sampler_type(&self) -> SamplerType {
    SamplerType::WeightOnly
  }

  fn sample_weight_only(&self, _: &RuntimeEnvironment, elems: Vec<f64>) -> Vec<usize> {
    aggregate_top_k_helper(elems.len(), self.k, |i| elems[i])
  }
}
