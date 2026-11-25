use std::cell::RefCell;
use std::collections::HashMap;

use oxidd::bcdd::{BCDDFunction, BCDDManagerRef};
use oxidd::util::{AllocResult, Ref};
use oxidd::{BooleanFunction, Function, HasLevel, Manager, ManagerRef, Node, NodeID};

use super::{BooleanFormula, Semiring};

const INITIAL_CAPACITY: usize = 2048;
const MAX_CAPACITY: usize = if usize::BITS > 32 {
  (1 << 31) - 1
} else {
  u32::MAX as usize / 24
};

thread_local! {
  static BUILDER: RefCell<Builder> = RefCell::new(Builder::new(INITIAL_CAPACITY));
}

pub(super) fn wmc<S: Semiring>(
  formula: &BooleanFormula,
  semiring: &S,
  weight: &impl Fn(&usize) -> S::Element,
) -> S::Element {
  let facts = formula.fact_ids();
  BUILDER.with(|slot| {
    if let Ok(mut builder) = slot.try_borrow_mut() {
      builder.count(formula, &facts, semiring, weight)
    } else {
      // A caller's weight function can itself request a count.
      Builder::new(INITIAL_CAPACITY).count(formula, &facts, semiring, weight)
    }
  })
}

struct Builder {
  manager: BCDDManagerRef,
  variables: Vec<BCDDFunction>,
  yes: BCDDFunction,
  no: BCDDFunction,
  capacity: usize,
}

impl Builder {
  fn new(capacity: usize) -> Self {
    let manager = oxidd::bcdd::new_manager(capacity, capacity / 2, 1);
    let (yes, no) = manager.with_manager_shared(|m| (BCDDFunction::t(m), BCDDFunction::f(m)));
    Self {
      manager,
      variables: Vec::new(),
      yes,
      no,
      capacity,
    }
  }

  fn count<S: Semiring>(
    &mut self,
    formula: &BooleanFormula,
    facts: &[usize],
    semiring: &S,
    weight: &impl Fn(&usize) -> S::Element,
  ) -> S::Element {
    let root = self.build(formula, facts);
    root.with_manager_shared(|manager, edge| {
      let yes = self.yes.as_edge(manager);
      let mut evaluation = Evaluation {
        terminal: yes.node_id(),
        true_tag: yes.tag(),
        facts,
        semiring,
        weight,
        cache: HashMap::new(),
      };
      evaluation.count(manager, edge)
    })
  }

  fn build(&mut self, formula: &BooleanFormula, facts: &[usize]) -> BCDDFunction {
    loop {
      if let Ok(root) = self.try_build(formula, facts) {
        return root;
      }
      if self.manager.with_manager_shared(|m| m.gc()) > 0
        && let Ok(root) = self.try_build(formula, facts)
      {
        return root;
      }
      assert!(
        self.capacity < MAX_CAPACITY,
        "OxiDD weighted model count exceeded node capacity"
      );
      *self = Self::new((self.capacity * 2).min(MAX_CAPACITY));
    }
  }

  fn try_build(&mut self, formula: &BooleanFormula, facts: &[usize]) -> AllocResult<BCDDFunction> {
    if facts.len() > self.variables.len() {
      let needed = u32::try_from(facts.len()).expect("too many variables for OxiDD weighted model counting");
      self.manager.with_manager_exclusive(|manager| {
        let registered = manager.num_vars();
        if registered < needed {
          manager.add_vars(needed - registered);
        }
        for id in self.variables.len() as u32..needed {
          self.variables.push(BCDDFunction::var(manager, id)?);
        }
        Ok(())
      })?;
    }
    self.compile(formula, facts)
  }

  fn compile(&self, formula: &BooleanFormula, facts: &[usize]) -> AllocResult<BCDDFunction> {
    match formula {
      BooleanFormula::Constant(true) => Ok(self.yes.clone()),
      BooleanFormula::Constant(false) => Ok(self.no.clone()),
      BooleanFormula::Literal(lit) => {
        let label = facts
          .binary_search(&lit.fact_id())
          .expect("fact was collected before compilation");
        let var = &self.variables[label];
        if lit.sign() {
          Ok(var.clone())
        } else {
          var.not()
        }
      }
      BooleanFormula::Not(term) => self.compile(term, facts)?.not(),
      BooleanFormula::And(terms) => terms
        .iter()
        .try_rfold(self.yes.clone(), |acc, term| self.compile(term, facts)?.and(&acc)),
      BooleanFormula::Or(terms) => terms
        .iter()
        .try_rfold(self.no.clone(), |acc, term| self.compile(term, facts)?.or(&acc)),
    }
  }
}

type BcddManager<'id> = <BCDDFunction as Function>::Manager<'id>;
type BcddEdge<'id> = <BcddManager<'id> as Manager>::Edge;

struct Evaluation<'a, 'id, S: Semiring, W> {
  terminal: NodeID,
  true_tag: <BcddManager<'id> as Manager>::EdgeTag,
  facts: &'a [usize],
  semiring: &'a S,
  weight: &'a W,
  cache: HashMap<(NodeID, bool), S::Element>,
}

impl<'id, S: Semiring, W: Fn(&usize) -> S::Element> Evaluation<'_, 'id, S, W> {
  fn constant(&self, edge: Ref<'_, BcddEdge<'id>>) -> Option<bool> {
    (edge.node_id() == self.terminal).then(|| edge.tag() == self.true_tag)
  }

  fn count<'a>(&mut self, manager: &'a BcddManager<'id>, edge: Ref<'a, BcddEdge<'id>>) -> S::Element {
    if let Some(value) = self.constant(edge) {
      return if value {
        self.semiring.one()
      } else {
        self.semiring.zero()
      };
    }
    let key = (edge.node_id(), edge.tag() == self.true_tag);
    if let Some(value) = self.cache.get(&key) {
      return value.clone();
    }
    let Node::Inner(node) = manager.get_node(edge) else {
      unreachable!("OxiDD BCDDs have only one terminal node");
    };
    let fact = self.facts[manager.level_to_var(node.level()) as usize];
    let (hi, lo) = BCDDFunction::cofactors_node(edge.tag(), node);
    let positive = (self.weight)(&fact);
    let semiring = self.semiring;
    let value = match (self.constant(hi), self.constant(lo)) {
      (Some(true), Some(false)) => return positive,
      (Some(false), Some(true)) => return semiring.negate(positive),
      (Some(false), _) => self.weighted_branch(manager, lo, semiring.negate(positive)),
      (_, Some(false)) => self.weighted_branch(manager, hi, positive),
      _ => {
        let negative = semiring.negate(positive.clone());
        let high = self.weighted_branch(manager, hi, positive);
        let low = self.weighted_branch(manager, lo, negative);
        semiring.add(high, low)
      }
    };
    self.cache.insert(key, value.clone());
    value
  }

  fn weighted_branch<'a>(
    &mut self,
    manager: &'a BcddManager<'id>,
    edge: Ref<'a, BcddEdge<'id>>,
    weight: S::Element,
  ) -> S::Element {
    if self.constant(edge) == Some(true) {
      weight
    } else {
      let count = self.count(manager, edge);
      self.semiring.mult(weight, count)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::runtime::provenance::{Literal, RealSemiring};

  #[test]
  fn grows_the_manager_when_a_formula_exceeds_capacity() {
    let mut builder = Builder::new(4);
    let formula = BooleanFormula::conjunction((0..8).map(|id| BooleanFormula::Literal(Literal::Pos(id))));
    let count = builder.count(&formula, &formula.fact_ids(), &RealSemiring, &|_| 0.5);
    assert_eq!(count, 1.0 / 256.0);
    assert!(builder.capacity > 4);
  }
}
