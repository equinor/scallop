use std::cmp::Reverse;
use std::collections::*;

use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

use super::{Clause, DNFFormula, Literal};

pub trait DNFContextTrait {
  fn fact_probability(&self, id: &usize) -> f64;

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool;

  fn literal_probability(&self, literal: &Literal) -> f64 {
    match literal {
      Literal::Pos(i) => self.fact_probability(i),
      Literal::Neg(i) => 1.0 - self.fact_probability(i),
    }
  }

  fn clause_probability(&self, clause: &Clause) -> f64 {
    clause
      .literals
      .iter()
      .fold(1.0, |agg, l| agg * self.literal_probability(l))
  }

  fn retain_no_conflict(&self, clauses: &mut Vec<Clause>) {
    clauses.retain(|proof| !self.has_disjunction_conflict(&proof.pos_fact_ids()))
  }

  fn retain_top_k(&self, clauses: &mut Vec<Clause>, k: usize) {
    if k == 0 {
      clauses.clear();
      return;
    }
    let mut ranked = clauses
      .drain(..)
      .map(|clause| (OrderedFloat(-self.clause_probability(&clause)), clause))
      .collect::<Vec<_>>();
    if k < ranked.len() {
      ranked.select_nth_unstable(k);
      ranked.truncate(k);
    }
    ranked.sort_unstable();
    clauses.extend(ranked.into_iter().map(|(_, clause)| clause));
  }

  fn sample_k_clauses<R: Rng + ?Sized>(&self, clauses: Vec<Clause>, k: usize, sampler: &mut R) -> Vec<Clause> {
    if clauses.is_empty() {
      vec![]
    } else if clauses.len() <= k {
      clauses
    } else {
      let weights = clauses.iter().map(|p| self.clause_probability(p)).collect::<Vec<_>>();
      if weights.iter().fold(0.0, |a, w| a + w) == 0.0 {
        clauses.into_iter().take(k).collect()
      } else {
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut sampled_ids = HashSet::new();
        let mut trial = 0; // Add trial count so that we don't fall into infinite loop
        while sampled_ids.len() < k && trial < k * 10 {
          let id = dist.sample(sampler);
          sampled_ids.insert(id);
          trial += 1;
        }
        clauses
          .into_iter()
          .enumerate()
          .filter_map(|(i, p)| if sampled_ids.contains(&i) { Some(p) } else { None })
          .collect()
      }
    }
  }

  fn cnf2dnf_k(&self, f: &[Clause], k: usize) -> DNFFormula {
    // Create a list of cnf clauses
    let cnf_clauses = f
      .iter()
      .sorted_unstable()
      .map(|clause| {
        clause
          .literals
          .iter()
          .sorted_by_cached_key(|&literal| (OrderedFloat(-self.literal_probability(literal)), literal))
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();

    let create_element = |indices: Vec<usize>| {
      // Collect the new set of literals from the `new_indices`. Note that there might not exist such a set
      // since the index can go over bound.
      //
      // We need to make sure that the literals are sorted by their variable id
      let mut literals: Vec<Literal> = (0..cnf_clauses.len())
        .map(|j| {
          if indices[j] < cnf_clauses[j].len() {
            Some(cnf_clauses[j][indices[j]].clone())
          } else {
            None
          }
        })
        .collect::<Option<Vec<_>>>()?;
      literals.sort_unstable_by_key(|l| (l.fact_id(), !l.sign()));
      literals.dedup();

      // Get the result
      let clause = Clause::new(literals);
      let prob = self.clause_probability(&clause);
      Some(Reverse((OrderedFloat(-prob), clause, indices)))
    };

    // A closure for getting the next elements of an element
    let next_elements = |indices: Vec<usize>| {
      // There will be `k` next elements if there are `k` clauses
      (0..indices.len()).filter_map(move |i| {
        // The `i`-th element will have an indices same as the current with only the `i`-th index incremented by 1.
        let new_indices = indices
          .iter()
          .enumerate()
          .map(|(j, x)| if i == j { *x + 1 } else { *x })
          .collect::<Vec<_>>();

        // Create the element using the generated indices
        create_element(new_indices)
      })
    };

    // Check corner cases
    if cnf_clauses.is_empty() {
      // If there is no clause, this is a "true"
      DNFFormula::one()
    } else if cnf_clauses.iter().any(|c| c.is_empty()) {
      // If there is a clause with no literal, this is a "false"
      DNFFormula::zero()
    } else {
      // Otherwise, perform full cartesian product
      let mut heap = BinaryHeap::new();
      let mut visited = HashSet::<Vec<usize>>::new();
      let mut result_clauses = Vec::new();

      // Get the first element
      let first_indices = vec![0; cnf_clauses.len()];
      let first_elem = create_element(first_indices).unwrap();

      // Push the element into the heap
      heap.push(first_elem);

      // Enter the main loop
      while result_clauses.len() < k {
        if let Some(Reverse((_, clause, indices))) = heap.pop() {
          if !visited.contains(&indices) {
            visited.insert(indices.clone());

            // Check validity of the clause; if valid, put it in the result
            if clause.is_valid() && !self.has_disjunction_conflict(&clause.pos_fact_ids()) {
              result_clauses.push(clause);
            }

            // Insert the successors into the heap
            for elem in next_elements(indices) {
              heap.push(elem);
            }
          } else {
            // Skip this clause
          }
        } else {
          break;
        }
      }

      // Return the result
      DNFFormula::new(result_clauses)
    }
  }

  fn base_zero(&self) -> DNFFormula {
    DNFFormula::zero()
  }

  fn base_one(&self) -> DNFFormula {
    DNFFormula::one()
  }

  fn base_add(&self, t1: &DNFFormula, t2: &DNFFormula) -> DNFFormula {
    t1.or(t2)
  }

  fn base_mult(&self, t1: &DNFFormula, t2: &DNFFormula) -> DNFFormula {
    t1.and(t2)
  }

  fn top_k_add(&self, t1: &DNFFormula, t2: &DNFFormula, k: usize) -> DNFFormula {
    let mut t = t1.or(t2);
    t.dedup();
    self.retain_top_k(&mut t.clauses, k);
    t
  }

  fn top_k_mult(&self, t1: &DNFFormula, t2: &DNFFormula, k: usize) -> DNFFormula {
    let mut t = t1.and(t2);
    t.dedup();
    self.retain_no_conflict(&mut t.clauses);
    self.retain_top_k(&mut t.clauses, k);
    t
  }

  fn top_k_negate(&self, t: &DNFFormula, k: usize) -> DNFFormula {
    // First generate a set of CNF clauses
    let cnf_clauses = t
      .clauses
      .iter()
      .map(|c| Clause::new(c.literals.iter().map(|l| l.negate()).collect()))
      .collect::<Vec<_>>();

    // Put these CNF clauses into cnf2dnf_k
    self.cnf2dnf_k(&cnf_clauses, k)
  }

  fn top_k_tag_of_chosen_set<'a, I>(&self, all: I, chosen_ids: &[usize], k: usize) -> DNFFormula
  where
    I: Iterator<Item = &'a DNFFormula>,
  {
    all
      .enumerate()
      .map(|(id, f)| {
        if chosen_ids.contains(&id) {
          f.clone()
        } else {
          self.top_k_negate(f, k)
        }
      })
      .fold(DNFFormula::one(), |a, b| self.top_k_mult(&a, &b, k))
  }
}
