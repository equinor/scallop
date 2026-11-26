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
  with_builder(|builder| {
    let root = builder.build(formula, &facts);
    count_root(&root, &builder.yes, &facts, semiring, weight)
  })
}

pub(crate) fn with_prepared_events<T>(
  lower: &BooleanFormula,
  upper: Option<&BooleanFormula>,
  evaluate: impl FnOnce(&PreparedWmc<'_>, Option<&PreparedWmc<'_>>) -> T,
) -> T {
  let lower_facts = lower.fact_ids();
  let upper_facts = upper.map(BooleanFormula::fact_ids);
  with_builder(|builder| {
    let lower = builder.prepare(lower, &lower_facts);
    let upper = upper
      .zip(upper_facts.as_deref())
      .map(|(formula, facts)| builder.prepare(formula, facts));
    evaluate(&lower, upper.as_ref())
  })
}

fn with_builder<T>(run: impl FnOnce(&mut Builder) -> T) -> T {
  BUILDER.with(|slot| {
    if let Ok(mut builder) = slot.try_borrow_mut() {
      run(&mut builder)
    } else {
      // A caller's weight function can itself request a count.
      run(&mut Builder::new(INITIAL_CAPACITY))
    }
  })
}

pub(crate) struct PreparedWmc<'a> {
  root: BCDDFunction,
  yes: BCDDFunction,
  facts: &'a [usize],
}

impl PreparedWmc<'_> {
  pub(crate) fn count<S: Semiring>(&self, semiring: &S, weight: &impl Fn(&usize) -> S::Element) -> S::Element {
    count_root(&self.root, &self.yes, self.facts, semiring, weight)
  }

  pub(crate) fn cofactors<S: Semiring>(
    &self,
    algebra: &S,
    weight: &impl Fn(&usize) -> S::Element,
  ) -> (S::Element, HashMap<usize, [S::Element; 2]>) {
    self.root.with_manager_shared(|manager, edge| {
      let yes = self.yes.as_edge(manager);
      let mut graph = CofactorGraph {
        terminal: yes.node_id(),
        true_tag: yes.tag(),
        variables: self.facts.len(),
        weights: self.facts.iter().map(weight).collect(),
        algebra,
        indices: HashMap::new(),
        nodes: vec![
          CofactorNode::terminal(self.facts.len(), algebra.zero()),
          CofactorNode::terminal(self.facts.len(), algebra.one()),
        ],
      };
      let root = graph.visit(manager, edge);
      let count = graph.nodes[root].suffix.clone();
      let cofactors = self.facts.iter().copied().zip(graph.cofactors(root)).collect();
      (count, cofactors)
    })
  }
}

struct CofactorNode<E> {
  variable: usize,
  children: [usize; 2],
  suffix: E,
}

impl<E> CofactorNode<E> {
  fn terminal(variable: usize, suffix: E) -> Self {
    Self {
      variable,
      children: [0, 0],
      suffix,
    }
  }
}

struct CofactorGraph<'a, 'id, S: Semiring> {
  terminal: NodeID,
  true_tag: <BcddManager<'id> as Manager>::EdgeTag,
  variables: usize,
  weights: Vec<S::Element>,
  algebra: &'a S,
  indices: HashMap<(NodeID, bool), usize>,
  nodes: Vec<CofactorNode<S::Element>>,
}

impl<'id, S: Semiring> CofactorGraph<'_, 'id, S> {
  fn visit<'a>(&mut self, manager: &'a BcddManager<'id>, edge: Ref<'a, BcddEdge<'id>>) -> usize {
    let positive = edge.tag() == self.true_tag;
    if edge.node_id() == self.terminal {
      return usize::from(positive);
    }
    let key = (edge.node_id(), positive);
    if let Some(&index) = self.indices.get(&key) {
      return index;
    }
    let Node::Inner(node) = manager.get_node(edge) else {
      unreachable!("OxiDD BCDDs have only one terminal node");
    };
    // Builders append variables and never reorder them, so variable IDs give
    // the dense order even when this root uses fewer variables than its manager.
    let variable = manager.level_to_var(node.level()) as usize;
    let (hi, lo) = BCDDFunction::cofactors_node(edge.tag(), node);
    let children = [self.visit(manager, lo), self.visit(manager, hi)];
    let [low, high] = self.branch_weights(variable);
    let suffix = self.algebra.add(
      self.algebra.mult(high, self.nodes[children[1]].suffix.clone()),
      self.algebra.mult(low, self.nodes[children[0]].suffix.clone()),
    );
    let index = self.nodes.len();
    self.nodes.push(CofactorNode {
      variable,
      children,
      suffix,
    });
    self.indices.insert(key, index);
    index
  }

  fn branch_weights(&self, variable: usize) -> [S::Element; 2] {
    let high = self.weights[variable].clone();
    [self.algebra.negate(high.clone()), high]
  }

  fn cofactors(&self, root: usize) -> Vec<[S::Element; 2]> {
    let a = self.algebra;
    let mut reach = vec![a.zero(); self.nodes.len()];
    reach[root] = a.one();
    let mut results = vec![[a.zero(), a.zero()]; self.variables];
    let size = self.variables.next_power_of_two();
    let mut bypass = vec![a.zero(); 2 * size];
    add_range(
      a,
      &mut bypass,
      size,
      0,
      self.nodes[root].variable,
      &self.nodes[root].suffix,
    );
    for (index, node) in self.nodes.iter().enumerate().skip(2).rev() {
      let variable = node.variable;
      for (branch, (child, weight)) in node.children.into_iter().zip(self.branch_weights(variable)).enumerate() {
        let suffix = &self.nodes[child].suffix;
        // When forced to this branch, the current variable supplies no weight.
        let result = &mut results[variable][branch];
        *result = a.add(result.clone(), a.mult(reach[index].clone(), suffix.clone()));
        let prefix = a.mult(reach[index].clone(), weight);
        reach[child] = a.add(reach[child].clone(), prefix.clone());
        let next = self.nodes[child].variable;
        if variable + 1 < next {
          // Successful paths skipping a variable contribute to both cofactors.
          let mass = a.mult(prefix, suffix.clone());
          add_range(a, &mut bypass, size, variable + 1, next, &mass);
        }
      }
    }
    for parent in 1..size {
      for child in [2 * parent, 2 * parent + 1] {
        bypass[child] = a.add(bypass[child].clone(), bypass[parent].clone());
      }
    }
    for (outputs, mass) in results.iter_mut().zip(bypass.into_iter().skip(size)) {
      for output in outputs {
        *output = a.add(output.clone(), mass.clone());
      }
    }
    results
  }
}

// Accumulate skipped-path mass without subtracting nearly equal probabilities.
fn add_range<S: Semiring>(a: &S, tree: &mut [S::Element], size: usize, start: usize, end: usize, value: &S::Element) {
  let (mut left, mut right) = (size + start, size + end);
  while left < right {
    if left % 2 == 1 {
      tree[left] = a.add(tree[left].clone(), value.clone());
      left += 1;
    }
    if right % 2 == 1 {
      right -= 1;
      tree[right] = a.add(tree[right].clone(), value.clone());
    }
    left /= 2;
    right /= 2;
  }
}

fn count_root<S: Semiring>(
  root: &BCDDFunction,
  yes: &BCDDFunction,
  facts: &[usize],
  semiring: &S,
  weight: &impl Fn(&usize) -> S::Element,
) -> S::Element {
  root.with_manager_shared(|manager, edge| {
    let yes = yes.as_edge(manager);
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

  fn prepare<'a>(&mut self, formula: &BooleanFormula, facts: &'a [usize]) -> PreparedWmc<'a> {
    let root = self.build(formula, facts);
    PreparedWmc {
      root,
      yes: self.yes.clone(),
      facts,
    }
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
  fn prepared_count_survives_manager_growth() {
    let mut builder = Builder::new(4);
    let earlier = BooleanFormula::Literal(Literal::Neg(19));
    let prepared = builder.prepare(&earlier, &[19]);
    let formula = BooleanFormula::conjunction((0..8).map(|id| BooleanFormula::Literal(Literal::Pos(id))));
    let facts = formula.fact_ids();
    let larger = builder.prepare(&formula, &facts);
    assert!(builder.capacity > 4);
    assert_eq!(larger.count(&RealSemiring, &|_| 0.5), 1.0 / 256.0);
    assert_eq!(
      prepared.count(&RealSemiring, &|id| if *id == 19 { 0.75 } else { 0.0 }),
      0.25
    );
    assert_eq!(prepared.count(&RealSemiring, &|_| 0.0), 1.0);
  }

  #[test]
  fn batch_cofactors_include_paths_that_skip_variables() {
    let lit = |id| BooleanFormula::Literal(Literal::Pos(id));
    let formula = BooleanFormula::disjunction([
      BooleanFormula::conjunction([lit(4), lit(19)]),
      BooleanFormula::conjunction([BooleanFormula::Literal(Literal::Neg(4)), lit(31)]),
    ]);
    let weights = |id: &usize| match id {
      4 => 0.75,
      19 => 0.0,
      31 => 0.5,
      _ => unreachable!(),
    };
    for (event, expected) in [(formula.clone(), false), (BooleanFormula::Not(Box::new(formula)), true)] {
      with_prepared_events(&event, None, |prepared, _| {
        let (mass, cofactors) = prepared.cofactors(&RealSemiring, &weights);
        assert_eq!(mass, if expected { 0.875 } else { 0.125 });
        for (&id, &[low, high]) in &cofactors {
          let forced = |value| {
            prepared.count(&RealSemiring, &|other| {
              if *other == id {
                value
              } else {
                weights(other)
              }
            })
          };
          assert_eq!([low, high], [forced(0.0), forced(1.0)]);
        }
      });
    }
  }
}
