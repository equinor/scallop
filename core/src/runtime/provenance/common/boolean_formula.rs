use std::collections::HashMap;
use std::ptr;

use rsdd::builder::sdd::CompressionSddBuilder;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{DDNNFPtr, SddPtr, VTree, VarLabel};

use super::Literal;

const DIRECT_WMC_MAX_FACTS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BooleanFormula {
  Constant(bool),
  Literal(Literal),
  Not(Box<Self>),
  And(Vec<Self>),
  Or(Vec<Self>),
}

impl BooleanFormula {
  pub fn conjunction(terms: impl IntoIterator<Item = Self>) -> Self {
    Self::combine(terms, true, Self::And)
  }

  pub fn disjunction(terms: impl IntoIterator<Item = Self>) -> Self {
    Self::combine(terms, false, Self::Or)
  }

  fn combine(terms: impl IntoIterator<Item = Self>, identity: bool, combine: impl FnOnce(Vec<Self>) -> Self) -> Self {
    let mut terms = terms.into_iter();
    let Some(first) = terms.next() else {
      return Self::Constant(identity);
    };
    let Some(second) = terms.next() else {
      return first;
    };
    combine([first, second].into_iter().chain(terms).collect())
  }

  pub(crate) fn wmc<S: Semiring>(&self, semiring: &S, weight: &impl Fn(&usize) -> S::Element) -> S::Element {
    match self {
      Self::Constant(true) => return semiring.one(),
      Self::Constant(false) => return semiring.zero(),
      Self::Literal(lit) => return literal_weight(semiring, weight, &lit.fact_id(), lit.sign()),
      _ => {}
    }

    // A flat clause of distinct facts can be counted directly, without allocating an SDD builder.
    if let Self::And(terms) | Self::Or(terms) = self
      && terms.len() > 1
      && terms
        .windows(2)
        .all(|pair| matches!(pair, [Self::Literal(a), Self::Literal(b)] if a.fact_id() < b.fact_id()))
    {
      if matches!(self, Self::And(_)) {
        return terms.iter().fold(semiring.one(), |acc, term| {
          semiring.mult(acc, term.wmc(semiring, weight))
        });
      }
      return terms.iter().rev().fold(semiring.zero(), |acc, term| {
        let Self::Literal(lit) = term else {
          unreachable!("flat clause contains only literals");
        };
        let positive = weight(&lit.fact_id());
        let negative = semiring.negate(positive.clone());
        let (yes, no) = if lit.sign() {
          (positive, negative)
        } else {
          (negative, positive)
        };
        semiring.add(yes, semiring.mult(no, acc))
      });
    }

    // RSDD's vtree is indexed by dense labels, not Scallop's potentially sparse fact IDs.
    let mut facts = Vec::new();
    self.collect_facts(&mut facts);
    facts.sort_unstable();
    facts.dedup();
    // Enumerate at most 256 assignments rather than allocating RSDD's fixed-size tables.
    if facts.len() <= DIRECT_WMC_MAX_FACTS {
      let weights: Vec<_> = facts
        .iter()
        .map(|fact| {
          let positive = weight(fact);
          let negative = semiring.negate(positive.clone());
          (positive, negative)
        })
        .collect();
      let matches: Vec<_> = (0..(1 << facts.len()))
        .map(|assignment| self.matches_assignment(&facts, assignment))
        .collect();
      return count_assignments(&matches, &weights, semiring).unwrap_or_else(|| semiring.zero());
    }

    let labels: Vec<_> = (0..facts.len()).map(VarLabel::new_usize).collect();
    let builder = CompressionSddBuilder::new(VTree::even_split(&labels, labels.len().ilog2() as usize));
    let root = self.compile(&builder, &facts);
    evaluate(root, &facts, semiring, weight, &mut HashMap::new())
  }

  fn matches_assignment(&self, facts: &[usize], assignment: usize) -> bool {
    match self {
      Self::Constant(value) => *value,
      Self::Literal(lit) => {
        let label = facts
          .binary_search(&lit.fact_id())
          .expect("fact was collected before counting");
        (assignment & (1 << label) != 0) == lit.sign()
      }
      Self::Not(term) => !term.matches_assignment(facts, assignment),
      Self::And(terms) => terms.iter().all(|term| term.matches_assignment(facts, assignment)),
      Self::Or(terms) => terms.iter().any(|term| term.matches_assignment(facts, assignment)),
    }
  }

  fn collect_facts(&self, facts: &mut Vec<usize>) {
    match self {
      Self::Constant(_) => {}
      Self::Literal(lit) => facts.push(lit.fact_id()),
      Self::Not(term) => term.collect_facts(facts),
      Self::And(terms) | Self::Or(terms) => {
        for term in terms {
          term.collect_facts(facts);
        }
      }
    }
  }

  fn compile<'a>(&self, builder: &'a CompressionSddBuilder<'a>, facts: &[usize]) -> SddPtr<'a> {
    match self {
      Self::Constant(true) => builder.true_ptr(),
      Self::Constant(false) => builder.false_ptr(),
      Self::Literal(lit) => {
        let label = facts
          .binary_search(&lit.fact_id())
          .expect("fact was collected before compilation");
        builder.var(VarLabel::new_usize(label), lit.sign())
      }
      Self::Not(term) => builder.negate(term.compile(builder, facts)),
      Self::And(terms) => terms.iter().fold(builder.true_ptr(), |acc, term| {
        builder.and(acc, term.compile(builder, facts))
      }),
      Self::Or(terms) => terms.iter().fold(builder.false_ptr(), |acc, term| {
        builder.or(acc, term.compile(builder, facts))
      }),
    }
  }
}

/// Weighted-counting operations. Elements may own data, such as sparse gradients.
pub trait Semiring {
  type Element: Clone;

  fn zero(&self) -> Self::Element;
  fn one(&self) -> Self::Element;
  fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element;
  fn mult(&self, a: Self::Element, b: Self::Element) -> Self::Element;
  fn negate(&self, a: Self::Element) -> Self::Element;
}

fn count_assignments<S: Semiring>(
  matches: &[bool],
  weights: &[(S::Element, S::Element)],
  semiring: &S,
) -> Option<S::Element> {
  let first = matches[0];
  if matches.iter().all(|&value| value == first) {
    return first.then(|| semiring.one());
  }
  let ((positive, negative), rest) = weights
    .split_last()
    .expect("nonconstant formula has an unassigned fact");
  let (negative_matches, positive_matches) = matches.split_at(matches.len() / 2);
  if negative_matches == positive_matches {
    return count_assignments(negative_matches, rest, semiring);
  }
  let yes = count_assignments(positive_matches, rest, semiring).map(|value| semiring.mult(positive.clone(), value));
  let no = count_assignments(negative_matches, rest, semiring).map(|value| semiring.mult(negative.clone(), value));
  match (yes, no) {
    (Some(yes), Some(no)) => Some(semiring.add(yes, no)),
    (yes, no) => yes.or(no),
  }
}

fn literal_weight<S: Semiring>(
  semiring: &S,
  weight: &impl Fn(&usize) -> S::Element,
  fact: &usize,
  positive: bool,
) -> S::Element {
  let value = weight(fact);
  if positive {
    value
  } else {
    semiring.negate(value)
  }
}

fn evaluate<S: Semiring>(
  node: SddPtr<'_>,
  facts: &[usize],
  semiring: &S,
  weight: &impl Fn(&usize) -> S::Element,
  cache: &mut HashMap<(*const (), bool), S::Element>,
) -> S::Element {
  let address = match node {
    SddPtr::PtrTrue => return semiring.one(),
    SddPtr::PtrFalse => return semiring.zero(),
    SddPtr::Var(label, positive) => return literal_weight(semiring, weight, &facts[label.value_usize()], positive),
    SddPtr::BDD(node) | SddPtr::ComplBDD(node) => ptr::from_ref(node).cast(),
    SddPtr::Reg(node) | SddPtr::Compl(node) => ptr::from_ref(node).cast(),
  };
  let complemented = node.is_neg();
  // Builder-owned nodes keep stable addresses for the entire traversal.
  let key = (address, complemented);
  if let Some(value) = cache.get(&key) {
    return value.clone();
  }
  // A complemented SDD negates each sub but leaves its partitioning primes intact.
  let value = node.node_iter().fold(semiring.zero(), |acc, pair| {
    let sub = if complemented { pair.sub.neg() } else { pair.sub };
    let prime_value = evaluate(pair.prime, facts, semiring, weight, cache);
    let sub_value = evaluate(sub, facts, semiring, weight, cache);
    semiring.add(acc, semiring.mult(prime_value, sub_value))
  });
  cache.insert(key, value.clone());
  value
}
