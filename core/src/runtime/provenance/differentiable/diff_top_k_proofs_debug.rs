use crate::common::foreign_tensor::*;
use crate::utils::*;

use super::*;

pub struct DiffTopKProofsDebugProvenance<T: FromTensor, P: PointerFamily, W: WmcBackend = DefaultWmcBackend> {
  pub k: usize,
  pub storage: DiffProbStorage<T, P>,
  pub disjunctions: P::Cell<Disjunctions>,
  pub wmc_with_disjunctions: bool,
  wmc_backend: W,
}

impl<T: FromTensor, P: PointerFamily, W: WmcBackend> Clone for DiffTopKProofsDebugProvenance<T, P, W> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      storage: self.storage.clone_internal(),
      disjunctions: P::clone_cell(&self.disjunctions),
      wmc_with_disjunctions: self.wmc_with_disjunctions,
      wmc_backend: self.wmc_backend.clone(),
    }
  }
}

impl<T: FromTensor, P: PointerFamily> DiffTopKProofsDebugProvenance<T, P> {
  pub fn new(k: usize, wmc_with_disjunctions: bool) -> Self {
    Self::new_with_wmc_backend(k, wmc_with_disjunctions, DefaultWmcBackend)
  }
}

impl<T: FromTensor, P: PointerFamily, W: WmcBackend> DiffTopKProofsDebugProvenance<T, P, W> {
  pub fn new_with_wmc_backend(k: usize, wmc_with_disjunctions: bool, wmc_backend: W) -> Self {
    Self {
      k,
      storage: DiffProbStorage::new(),
      disjunctions: P::new_cell(Disjunctions::new()),
      wmc_with_disjunctions,
      wmc_backend,
    }
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.storage.input_tags()
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }
}

impl<T: FromTensor, P: PointerFamily, W: WmcBackend> DNFContextTrait for DiffTopKProofsDebugProvenance<T, P, W> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.storage.fact_probability(id)
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    P::get_cell(&self.disjunctions, |d| d.has_conflict(pos_facts))
  }
}

impl<T: FromTensor, P: PointerFamily, W: WmcBackend> Provenance for DiffTopKProofsDebugProvenance<T, P, W> {
  type Tag = DNFFormula;

  type InputTag = InputExclusiveDiffProbWithID<T>;

  type OutputTag = OutputDiffProbWithProofs;

  fn name(&self) -> String {
    "diff-top-k-proofs".to_string()
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    let InputExclusiveDiffProbWithID {
      prob,
      id,
      external_tag,
      exclusion,
    } = input_tag;

    // First store the probability and generate the id
    let fact_id = self.storage.add_prob_with_id(prob, external_tag, id);

    // Store the mutual exclusivity
    if let Some(disjunction_id) = exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disjunction_id, fact_id));
    }

    // Finally return the formula
    DNFFormula::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    // Get the number of variables that requires grad
    let num_var_requires_grad = self.storage.num_input_tags();
    let s = DualNumberSemiring::new(num_var_requires_grad + 1);
    let v = |i: &usize| {
      let (real, external_tag) = self.storage.get_diff_prob(i);

      // Check if this variable `i` requires grad or not
      if external_tag.is_some() {
        s.singleton(real, *i)
      } else {
        s.constant(real)
      }
    };
    let wmc_result = if self.wmc_with_disjunctions {
      P::get_cell(&self.disjunctions, |disj| {
        t.wmc_with_disjunctions_using(&self.wmc_backend, &s, &v, disj)
      })
    } else {
      t.wmc_using(&self.wmc_backend, &s, &v)
    };
    let prob = wmc_result.real;
    let deriv = wmc_result
      .deriv
      .iter()
      .map(|(id, weight)| (id, *weight))
      .collect::<Vec<_>>();

    let proofs = t
      .iter()
      .map(|clause| {
        clause
          .iter()
          .map(|literal| match literal {
            Literal::Pos(id) => (true, *id),
            Literal::Neg(id) => (false, *id),
          })
          .collect()
      })
      .collect();

    OutputDiffProbWithProofs {
      probability: prob,
      gradient: deriv,
      proofs,
    }
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
    self.top_k_add(t1, t2, self.k)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_mult(t1, t2, self.k)
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.top_k_negate(t, self.k))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let v = |i: &usize| self.storage.get_prob(i);
    t.wmc_using(&self.wmc_backend, &RealSemiring::new(), &v)
  }
}
