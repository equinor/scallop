use std::rc::Rc;

use super::*;
use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::common::tuple_access::TupleAccessor;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, Prov: Provenance> {
  source: DynamicDataflow<'a, Prov>,
  projection: Rc<Projection>,
  runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicProjectDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_stable())
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.apply(self.source.iter_recent())
  }
}

impl<'a, Prov: Provenance> DynamicProjectDataflow<'a, Prov> {
  pub fn new(source: DynamicDataflow<'a, Prov>, expression: Expr, runtime: &'a RuntimeEnvironment) -> Self {
    let projection = OwnedProjection::compile(&expression, &mut Vec::new())
      .map_or_else(|| Projection::General(expression), Projection::Owned);
    Self {
      source,
      projection: Rc::new(projection),
      runtime,
    }
  }

  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let runtime = self.runtime;
    let projection = Rc::clone(&self.projection);
    DynamicBatches::from_iterator(batches.map(move |batch| {
      let projection = Rc::clone(&projection);
      DynamicBatch::from_iterator(batch.filter_map(move |mut elem| {
        elem.tuple = match &*projection {
          Projection::Owned(projection) => projection.eval(&mut elem.tuple),
          Projection::General(expr) => runtime.eval(expr, &elem.tuple)?,
        };
        Some(elem)
      }))
    }))
  }
}

enum Projection {
  General(Expr),
  Owned(OwnedProjection),
}

enum OwnedProjection {
  Access { path: TupleAccessor, take: bool },
  Tuple(Vec<Self>),
}

impl OwnedProjection {
  fn compile(expr: &Expr, later: &mut Vec<TupleAccessor>) -> Option<Self> {
    match expr {
      Expr::Tuple(exprs) => {
        let mut fields = exprs
          .iter()
          .rev()
          .map(|expr| Self::compile(expr, later))
          .collect::<Option<Vec<_>>>()?;
        fields.reverse();
        Some(Self::Tuple(fields))
      }
      Expr::Access(path) => {
        let indices = path.indices.get(..path.len())?;
        // A later read of this path, its parent, or a child needs the original value.
        let take = !later.iter().any(|other| {
          let prefix_len = path.len().min(other.len());
          indices[..prefix_len] == other.indices[..prefix_len]
        });
        later.push(*path);
        Some(Self::Access { path: *path, take })
      }
      _ => None,
    }
  }

  fn eval(&self, input: &mut Tuple) -> Tuple {
    match self {
      Self::Access { path, take: false } => input[path].clone(),
      Self::Access { path, take: true } => take_field(input, *path),
      Self::Tuple(fields) => Tuple::tuple(fields.iter().map(|field| field.eval(input))),
    }
  }
}

fn take_field(input: &mut Tuple, path: TupleAccessor) -> Tuple {
  match (input, path.len) {
    (input, 0) => std::mem::replace(input, Tuple::unit()),
    (Tuple::Tuple(fields), _) => take_field(&mut fields[path.indices[0] as usize], path.shift()),
    _ => panic!("Cannot access tuple with `{:?}`", path),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::value::Value;

  #[test]
  fn pure_projection_moves_last_uses_and_keeps_overlapping_reads_intact() {
    fn string_ptr(tuple: &Tuple) -> *const u8 {
      let Some(Value::String(value)) = tuple.get_value() else {
        panic!("expected a string");
      };
      value.as_ptr()
    }

    let mut input = Tuple::from(((String::from("first"), String::from("second")), String::from("third")));
    let second = string_ptr(&input[0][1]);
    let third = string_ptr(&input[1]);
    let expr = Expr::from((
      Expr::access(0),
      (Expr::access((0, 1)), Expr::access((0, 1))),
      Expr::access(1),
    ));
    let expected = RuntimeEnvironment::default().eval(&expr, &input).unwrap();
    let plan = OwnedProjection::compile(&expr, &mut Vec::new()).unwrap();
    let projected = plan.eval(&mut input);
    assert_eq!(projected, expected);
    assert_ne!(string_ptr(&projected[0][1]), second);
    assert_ne!(string_ptr(&projected[1][0]), second);
    assert_eq!(string_ptr(&projected[1][1]), second);
    assert_eq!(string_ptr(&projected[2]), third);
  }
}
