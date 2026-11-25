use std::simd::cmp::SimdPartialOrd;
use std::simd::{Simd, SimdElement};

use super::*;
use crate::common::binary_op::BinaryOp;
use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::common::value::Value;

#[derive(Clone)]
pub struct DynamicFilterDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub filter: Expr,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicFilterDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicFilterDataflow<'a, Prov> {
  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let runtime = self.runtime;
    let filter = self.filter.clone();
    let packed = IntegerFilter::from_expr(&filter);
    DynamicBatches::from_iterator(batches.map(move |batch| {
      let filter = filter.clone();
      if let Some(predicate) = packed
        && batch.remaining_slice().is_some_and(|rows| rows.len() >= LANES)
      {
        DynamicBatch::new(PackedFilterBatch {
          source: batch,
          predicate,
          filter,
          runtime,
          matches: 0,
          next: 0,
        })
      } else {
        batch.filter(move |elem| eval_filter(runtime, &filter, &elem.tuple))
      }
    }))
  }
}

fn eval_filter(runtime: &RuntimeEnvironment, filter: &Expr, tuple: &Tuple) -> bool {
  runtime.eval(filter, tuple).is_some_and(|result| result.as_bool())
}

const LANES: usize = 8;

#[derive(Clone, Copy)]
enum Comparison {
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
}

impl Comparison {
  fn from_op(op: &BinaryOp, reversed: bool) -> Option<Self> {
    match (op, reversed) {
      (BinaryOp::Eq, _) => Some(Self::Eq),
      (BinaryOp::Neq, _) => Some(Self::Neq),
      (BinaryOp::Lt, false) | (BinaryOp::Gt, true) => Some(Self::Lt),
      (BinaryOp::Leq, false) | (BinaryOp::Geq, true) => Some(Self::Leq),
      (BinaryOp::Gt, false) | (BinaryOp::Lt, true) => Some(Self::Gt),
      (BinaryOp::Geq, false) | (BinaryOp::Leq, true) => Some(Self::Geq),
      _ => None,
    }
  }

  fn compare<V: SimdPartialOrd>(self, values: V, rhs: V) -> V::Mask {
    match self {
      Self::Eq => values.simd_eq(rhs),
      Self::Neq => values.simd_ne(rhs),
      Self::Lt => values.simd_lt(rhs),
      Self::Leq => values.simd_le(rhs),
      Self::Gt => values.simd_gt(rhs),
      Self::Geq => values.simd_ge(rhs),
    }
  }

  fn compare_scalar<T: Ord>(self, lhs: T, rhs: T) -> bool {
    match self {
      Self::Eq => lhs == rhs,
      Self::Neq => lhs != rhs,
      Self::Lt => lhs < rhs,
      Self::Leq => lhs <= rhs,
      Self::Gt => lhs > rhs,
      Self::Geq => lhs >= rhs,
    }
  }
}

#[derive(Clone, Copy)]
enum IntegerConstant {
  I32(i32),
  I64(i64),
}

#[derive(Clone, Copy)]
struct IntegerFilter {
  field: usize,
  op: Comparison,
  rhs: IntegerConstant,
}

impl IntegerFilter {
  fn from_expr(expr: &Expr) -> Option<Self> {
    let Expr::Binary(expr) = expr else {
      return None;
    };
    let (accessor, constant, reversed) = match (&*expr.op1, &*expr.op2) {
      (Expr::Access(accessor), Expr::Constant(value)) => (accessor, value, false),
      (Expr::Constant(value), Expr::Access(accessor)) => (accessor, value, true),
      _ => return None,
    };
    if accessor.len() != 1 {
      return None;
    }
    let field = usize::try_from(accessor.indices[0]).ok()?;
    let op = Comparison::from_op(&expr.op, reversed)?;
    let rhs = match constant {
      Value::I32(n) => IntegerConstant::I32(*n),
      Value::I64(n) => IntegerConstant::I64(*n),
      _ => return None,
    };
    Some(Self { field, op, rhs })
  }

  fn field_value(self, tuple: &Tuple) -> Option<&Value> {
    let Tuple::Tuple(fields) = tuple else {
      return None;
    };
    fields.get(self.field)?.get_value()
  }

  fn pack<T: Default + SimdElement, P: Provenance>(
    self,
    rows: &[DynamicElement<P>; LANES],
    extract: impl Fn(&Value) -> Option<T>,
  ) -> Option<Simd<T, LANES>> {
    let mut values = [T::default(); LANES];
    for (out, row) in values.iter_mut().zip(rows) {
      *out = extract(self.field_value(&row.tuple)?)?;
    }
    Some(Simd::from_array(values))
  }

  fn mask<P: Provenance>(self, rows: &[DynamicElement<P>; LANES]) -> Option<u64> {
    match self.rhs {
      IntegerConstant::I32(rhs) => {
        let values = self.pack(rows, |value| match value {
          Value::I32(n) => Some(*n),
          _ => None,
        })?;
        Some(self.op.compare(values, Simd::splat(rhs)).to_bitmask())
      }
      IntegerConstant::I64(rhs) => {
        let values = self.pack(rows, |value| match value {
          Value::I64(n) => Some(*n),
          _ => None,
        })?;
        Some(self.op.compare(values, Simd::splat(rhs)).to_bitmask())
      }
    }
  }

  fn matches_scalar(self, tuple: &Tuple) -> Option<bool> {
    match (self.field_value(tuple)?, self.rhs) {
      (Value::I32(lhs), IntegerConstant::I32(rhs)) => Some(self.op.compare_scalar(*lhs, rhs)),
      (Value::I64(lhs), IntegerConstant::I64(rhs)) => Some(self.op.compare_scalar(*lhs, rhs)),
      _ => None,
    }
  }
}

#[derive(Clone)]
struct PackedFilterBatch<'a, P: Provenance> {
  source: DynamicBatch<'a, P>,
  predicate: IntegerFilter,
  filter: Expr,
  runtime: &'a RuntimeEnvironment,
  matches: u64,
  next: usize,
}

impl<'a, P: Provenance> Batch<'a, P> for PackedFilterBatch<'a, P> {
  fn next_elem(&mut self) -> Option<DynamicElement<P>> {
    let rows = self
      .source
      .remaining_slice()
      .expect("packed filter needs contiguous input");
    loop {
      if self.matches != 0 {
        let lane = self.matches.trailing_zeros() as usize;
        self.matches &= self.matches - 1;
        return Some(rows[self.next - LANES + lane].clone());
      }

      let remaining = &rows[self.next..];
      if let Some(chunk) = remaining.first_chunk()
        && let Some(matches) = self.predicate.mask(chunk)
      {
        // Keep the mask across calls so rejected rows never need to be cloned.
        self.matches = matches;
        self.next += LANES;
      } else {
        // Preserve the evaluator's behavior on unexpected tuple shapes/types.
        let found = remaining.iter().position(|row| {
          self
            .predicate
            .matches_scalar(&row.tuple)
            .unwrap_or_else(|| eval_filter(self.runtime, &self.filter, &row.tuple))
        });
        return if let Some(i) = found {
          self.next += i + 1;
          Some(remaining[i].clone())
        } else {
          self.next = rows.len();
          None
        };
      }
    }
  }
}
