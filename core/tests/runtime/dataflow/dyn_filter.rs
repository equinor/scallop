use scallop_core::common::binary_op::BinaryOp;
use scallop_core::common::expr::*;
use scallop_core::common::tuple::Tuple;
use scallop_core::common::value::Value;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_filter_1() {
  let ctx = unit::UnitProvenance;
  let rt = RuntimeEnvironment::default();

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&ctx, vec![(0i8, 1i8), (1i8, 2i8)]);

  // Iterate until fixpoint
  while source.changed(&ctx, rt.get_default_scheduler()) || target.changed(&ctx, rt.get_default_scheduler()) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::filter(
        DynamicDataflow::dynamic_relation(&source),
        Expr::access(1).gt(Expr::constant(1i8)),
        &rt,
      ),
      &rt,
    )
  }

  expect_collection(&source.complete(&ctx).into(), vec![(0i8, 1i8), (1i8, 2i8)]);
  expect_collection(&target.complete(&ctx).into(), vec![(1i8, 2i8)]);
}

#[test]
fn test_packed_integer_filters_match_scalar_order_and_tags() {
  let rt = RuntimeEnvironment::default();
  for (values, constant) in [
    (
      [Value::I32(i32::MIN), Value::I32(-7), Value::I32(i32::MAX)],
      Value::I32(-7),
    ),
    (
      [Value::I64(i64::MIN), Value::I64(-7), Value::I64(i64::MAX)],
      Value::I64(-7),
    ),
  ] {
    let rows = DynamicSortedCollection::<natural::NaturalProvenance>::from_vec_unchecked(
      (0..27)
        .map(|i| {
          let value = match i {
            0..=7 => &values[0],
            16..=23 => &values[2],
            _ => &values[i % 3],
          };
          DynamicElement::new((i as i32, value.clone()), i + 1)
        })
        .collect(),
    );
    let relation = DynamicRelation::new();
    *relation.recent.borrow_mut() = rows.clone();
    relation.stable.borrow_mut().push(rows.clone());
    for op in [
      BinaryOp::Eq,
      BinaryOp::Neq,
      BinaryOp::Lt,
      BinaryOp::Leq,
      BinaryOp::Gt,
      BinaryOp::Geq,
    ] {
      for reversed in [false, true] {
        let access = Expr::access(1);
        let constant = Expr::constant(constant.clone());
        let (lhs, rhs) = if reversed {
          (constant, access)
        } else {
          (access, constant)
        };
        let expr = Expr::binary(op.clone(), lhs, rhs);
        let expected = rows
          .elements
          .iter()
          .filter(|row| rt.eval(&expr, &row.tuple).unwrap().as_bool())
          .map(|row| (row.tuple.clone(), row.tag))
          .collect::<Vec<_>>();
        for recent in [true, false] {
          let sources = [
            DynamicDataflow::dynamic_collection(DynamicCollectionRef::Sorted(&rows), recent),
            DynamicDataflow::dynamic_relation(&relation),
          ];
          for source in sources {
            let filter = source.filter(expr.clone(), &rt);
            let mut batches = if recent {
              filter.iter_recent()
            } else {
              filter.iter_stable()
            };
            let mut batch = batches.next().unwrap();
            assert_eq!(batch.next().map(|row| (row.tuple, row.tag)).as_ref(), expected.first());
            assert!(batch
              .clone()
              .map(|row| (row.tuple, row.tag))
              .eq(expected.iter().skip(1).cloned()));
            assert!(batch
              .map(|row| (row.tuple, row.tag))
              .eq(expected.iter().skip(1).cloned()));
          }
        }
      }
    }
  }
}

#[test]
fn test_packed_integer_filter_preserves_scalar_error_position() {
  let rt = RuntimeEnvironment::default();
  let rows: Vec<DynamicElement<unit::UnitProvenance>> = (0..9)
    .map(|i| {
      let value = if i == 3 { Value::I64(5) } else { Value::I32(i) };
      DynamicElement::new((value,), unit::Unit)
    })
    .collect();
  let filter = DynamicDataflow::vec(&rows).filter(Expr::access(0).gt(Expr::constant(1i32)), &rt);
  let mut batch = filter.iter_recent().next().unwrap();
  assert_eq!(batch.next().unwrap().tuple, Tuple::from((2i32,)));
  assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| batch.next())).is_err());
}
