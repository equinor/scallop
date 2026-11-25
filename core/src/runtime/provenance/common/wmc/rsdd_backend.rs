use std::collections::HashMap;
use std::ptr;

use rsdd::builder::sdd::CompressionSddBuilder;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{DDNNFPtr, SddPtr, VTree, VarLabel};

use super::{BooleanFormula, Semiring};

pub(super) fn wmc<S: Semiring>(
  formula: &BooleanFormula,
  semiring: &S,
  weight: &impl Fn(&usize) -> S::Element,
) -> S::Element {
  let facts = formula.fact_ids();
  // An SDD vtree needs a leaf even for a constant formula.
  let labels: Vec<_> = (0..facts.len().max(1)).map(VarLabel::new_usize).collect();
  let builder = CompressionSddBuilder::new(VTree::even_split(&labels, labels.len().ilog2() as usize));
  evaluate(
    compile(formula, &builder, &facts),
    &facts,
    semiring,
    weight,
    &mut HashMap::new(),
  )
}

fn compile<'a>(formula: &BooleanFormula, builder: &'a CompressionSddBuilder<'a>, facts: &[usize]) -> SddPtr<'a> {
  match formula {
    BooleanFormula::Constant(true) => builder.true_ptr(),
    BooleanFormula::Constant(false) => builder.false_ptr(),
    BooleanFormula::Literal(lit) => {
      let label = facts
        .binary_search(&lit.fact_id())
        .expect("fact was collected before compilation");
      builder.var(VarLabel::new_usize(label), lit.sign())
    }
    BooleanFormula::Not(term) => builder.negate(compile(term, builder, facts)),
    BooleanFormula::And(terms) => terms.iter().fold(builder.true_ptr(), |acc, term| {
      builder.and(acc, compile(term, builder, facts))
    }),
    BooleanFormula::Or(terms) => terms.iter().fold(builder.false_ptr(), |acc, term| {
      builder.or(acc, compile(term, builder, facts))
    }),
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
    SddPtr::Var(label, positive) => {
      let value = weight(&facts[label.value_usize()]);
      return if positive { value } else { semiring.negate(value) };
    }
    SddPtr::BDD(node) | SddPtr::ComplBDD(node) => ptr::from_ref(node).cast(),
    SddPtr::Reg(node) | SddPtr::Compl(node) => ptr::from_ref(node).cast(),
  };
  let complemented = node.is_neg();
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
