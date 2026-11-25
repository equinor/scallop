use std::rc::Rc;

use super::*;
use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::common::tuple_access::TupleAccessor;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, Prov: Provenance> {
  source: DynamicDataflow<'a, Prov>,
  projection: Rc<Projection>,
  preserves_order: bool,
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
    let preserves_order = expression.is_order_preserving_projection();
    let projection = OwnedProjection::compile(&expression, &mut Vec::new())
      .map_or_else(|| Projection::General(expression), Projection::Owned);
    Self {
      source,
      projection: Rc::new(projection),
      preserves_order,
      runtime,
    }
  }

  fn apply(&self, batches: DynamicBatches<'a, Prov>) -> DynamicBatches<'a, Prov> {
    let runtime = self.runtime;
    let projection = Rc::clone(&self.projection);
    let preserves_order = self.preserves_order;
    DynamicBatches::from_iterator(batches.map(move |source| {
      DynamicBatch::new(ProjectBatch {
        sorted: preserves_order && source.is_tuple_sorted(),
        source,
        projection: Rc::clone(&projection),
        runtime,
      })
    }))
  }
}

#[derive(Clone)]
struct ProjectBatch<'a, Prov: Provenance> {
  source: DynamicBatch<'a, Prov>,
  projection: Rc<Projection>,
  sorted: bool,
  runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ProjectBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.source.find_map(|mut elem| {
      elem.tuple = match &*self.projection {
        Projection::Owned(projection) => projection.eval(&mut elem.tuple),
        Projection::General(expr) => self.runtime.eval(expr, &elem.tuple)?,
      };
      Some(elem)
    })
  }

  fn is_tuple_sorted(&self) -> bool {
    self.sorted
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
  use crate::runtime::provenance::natural::NaturalProvenance;

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

  #[test]
  fn prefix_projection_preserves_batch_order_and_duplicate_tags() {
    let rt = RuntimeEnvironment::default();
    let rows = DynamicSortedCollection::<NaturalProvenance>::from_vec_unchecked(vec![
      DynamicElement::new((1i32, (2i32, 4i32), 5i32), 2),
      DynamicElement::new((1i32, (2i32, 4i32), 6i32), 3),
      DynamicElement::new((2i32, (0i32, 1i32), 6i32), 4),
    ]);
    let sorted = DynamicDataflow::dynamic_collection(DynamicCollectionRef::Sorted(&rows), true);
    let prefix = Expr::from((Expr::access(0), (Expr::access(1),)));
    let projected = sorted
      .clone()
      .filter(Expr::access(2).gt(Expr::constant(4i32)), &rt)
      .project(prefix.clone(), &rt);
    let batch = projected.iter_recent().next().unwrap();
    assert!(batch.is_tuple_sorted());
    assert!(batch.map(|row| (row.tuple, row.tag)).eq([
      (Tuple::from((1i32, ((2i32, 4i32),))), 2),
      (Tuple::from((1i32, ((2i32, 4i32),))), 3),
      (Tuple::from((2i32, ((0i32, 1i32),))), 4),
    ]));

    let suffix = sorted.project(Expr::access(1), &rt);
    let batch = suffix.iter_recent().next().unwrap();
    assert!(!batch.is_tuple_sorted());
    assert!(batch.map(|row| row.tuple).eq([
      Tuple::from((2i32, 4i32)),
      Tuple::from((2i32, 4i32)),
      Tuple::from((0i32, 1i32)),
    ]));

    let unsorted = DynamicDataflow::vec(&rows.elements).project(prefix, &rt);
    assert!(!unsorted.iter_recent().next().unwrap().is_tuple_sorted());
  }
}
