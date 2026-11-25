use std::collections::*;

use rand::prelude::*;

use crate::utils::*;

use super::*;

pub struct SampleKProofsProvenance<P: PointerFamily = RcFamily, W: WmcBackend = DefaultWmcBackend> {
  pub k: usize,
  pub sampler: P::Cell<SmallRng>,
  pub probs: P::Cell<Vec<f64>>,
  pub disjunctions: P::Cell<Disjunctions>,
  wmc_backend: W,
}

impl<P: PointerFamily, W: WmcBackend> Clone for SampleKProofsProvenance<P, W> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      sampler: P::clone_cell(&self.sampler),
      probs: P::clone_cell(&self.probs),
      disjunctions: P::clone_cell(&self.disjunctions),
      wmc_backend: self.wmc_backend.clone(),
    }
  }
}

impl<P: PointerFamily> SampleKProofsProvenance<P> {
  pub fn new(k: usize) -> Self {
    Self::new_with_seed(k, 12345678)
  }

  pub fn new_with_seed(k: usize, seed: u64) -> Self {
    Self::new_with_wmc_backend(k, seed, DefaultWmcBackend)
  }
}

impl<P: PointerFamily, W: WmcBackend> SampleKProofsProvenance<P, W> {
  pub fn new_with_wmc_backend(k: usize, seed: u64, wmc_backend: W) -> Self {
    Self {
      k,
      sampler: P::new_cell(SmallRng::seed_from_u64(seed)),
      probs: P::new_cell(Vec::new()),
      disjunctions: P::new_cell(Disjunctions::new()),
      wmc_backend,
    }
  }

  pub fn set_k(&mut self, new_k: usize) {
    self.k = new_k;
  }
}

impl<P: PointerFamily, W: WmcBackend> DNFContextTrait for SampleKProofsProvenance<P, W> {
  fn fact_probability(&self, id: &usize) -> f64 {
    P::get_cell(&self.probs, |p| p[*id])
  }

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool {
    P::get_cell(&self.disjunctions, |d| d.has_conflict(pos_facts))
  }
}

impl<P: PointerFamily, W: WmcBackend> Provenance for SampleKProofsProvenance<P, W> {
  type Tag = DNFFormula;

  type InputTag = InputExclusiveProb;

  type OutputTag = f64;

  fn name(&self) -> String {
    "sample-k-proofs".to_string()
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    // First generate id and push the probability into the list
    let fact_id = P::get_cell(&self.probs, |p| p.len());
    P::get_cell_mut(&self.probs, |p| p.push(input_tag.prob));

    // Add exlusion if needed
    if let Some(disj_id) = input_tag.exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disj_id, fact_id));
    }

    // Lastly return a tag
    Self::Tag::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    t.wmc_using(&self.wmc_backend, &s, &v)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    self.base_zero()
  }

  fn one(&self) -> Self::Tag {
    self.base_one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    let tag = t1.or(t2);
    let sampled_clauses = P::get_cell_mut(&self.sampler, |s| self.sample_k_clauses(tag.clauses, self.k, s));
    DNFFormula {
      clauses: sampled_clauses,
    }
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    let mut tag = t1.or(t2);
    self.retain_no_conflict(&mut tag.clauses);
    let sampled_clauses = P::get_cell_mut(&self.sampler, |s| self.sample_k_clauses(tag.clauses, self.k, s));
    DNFFormula {
      clauses: sampled_clauses,
    }
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }

  fn minus(&self, _: &Self::Tag, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    t.wmc_using(&self.wmc_backend, &s, &v)
  }
}
