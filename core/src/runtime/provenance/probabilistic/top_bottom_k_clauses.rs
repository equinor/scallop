use std::collections::*;

use super::*;
use crate::utils::{PointerFamily, RcFamily};

#[derive(Debug)]
pub struct TopBottomKClausesProvenance<P: PointerFamily = RcFamily, W: WmcBackend = DefaultWmcBackend> {
  pub k: usize,
  pub probs: P::Cell<Vec<f64>>,
  pub disjunctions: P::Cell<Disjunctions>,
  wmc_backend: W,
}

impl<P: PointerFamily, W: WmcBackend> Clone for TopBottomKClausesProvenance<P, W> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      probs: P::clone_cell(&self.probs),
      disjunctions: P::clone_cell(&self.disjunctions),
      wmc_backend: self.wmc_backend.clone(),
    }
  }
}

impl<P: PointerFamily> TopBottomKClausesProvenance<P> {
  pub fn new(k: usize) -> Self {
    Self::new_with_wmc_backend(k, DefaultWmcBackend)
  }
}

impl<P: PointerFamily, W: WmcBackend> TopBottomKClausesProvenance<P, W> {
  pub fn new_with_wmc_backend(k: usize, wmc_backend: W) -> Self {
    Self {
      k,
      probs: P::new_cell(Vec::new()),
      disjunctions: P::new_cell(Disjunctions::new()),
      wmc_backend,
    }
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }
}

impl<P: PointerFamily, W: WmcBackend> CNFDNFContextTrait for TopBottomKClausesProvenance<P, W> {
  fn fact_probability(&self, id: &usize) -> f64 {
    P::get_cell(&self.probs, |p| p[*id])
  }

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool {
    P::get_cell(&self.disjunctions, |d| d.has_conflict(pos_facts))
  }
}

impl<P: PointerFamily, W: WmcBackend> Provenance for TopBottomKClausesProvenance<P, W> {
  type Tag = CNFDNFFormula;

  type InputTag = InputExclusiveProb;

  type OutputTag = f64;

  fn name(&self) -> String {
    "top-bottom-k-clauses".to_string()
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    // First generate id and push the probability into the list
    let fact_id = P::get_cell(&self.probs, |p| p.len());
    P::get_cell_mut(&self.probs, |p| p.push(input_tag.prob));

    // Add exlusion if needed
    if let Some(disj_id) = input_tag.exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disj_id, fact_id));
    }

    // Return the formula
    CNFDNFFormula::dnf_singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    t.wmc_using(&self.wmc_backend, &s, &v)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_zero()
  }

  fn zero(&self) -> Self::Tag {
    CNFDNFFormula::dnf_zero()
  }

  fn one(&self) -> Self::Tag {
    CNFDNFFormula::dnf_one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_bottom_k_add(t1, t2, self.k)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_bottom_k_mult(t1, t2, self.k)
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.base_negate(t))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    t.wmc_using(&self.wmc_backend, &s, &v)
  }
}
