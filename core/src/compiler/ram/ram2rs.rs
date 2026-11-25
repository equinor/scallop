use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};
use std::collections::*;

use super::*;
use crate::common::adt_variant_registry::ADTVariantRegistry;
use crate::common::expr::*;
use crate::common::foreign_aggregate::{AggregateInfo, AggregateRegistry};
use crate::common::foreign_function::ForeignFunctionRegistry;
use crate::common::foreign_predicate::ForeignPredicateRegistry;
use crate::common::input_file::InputFile;
use crate::common::input_tag::*;
use crate::common::output_option::{OutputFile, OutputOption};
use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::common::unary_op::UnaryOp;
use crate::common::value::*;
use crate::common::value_type::*;
use crate::compiler::options::CompileOptions;
use crate::runtime::database::StorageMetadata;
use crate::runtime::env::Scheduler;

#[derive(Debug, thiserror::Error)]
pub enum RustEmissionError {
  #[error("Rust code generation does not support custom foreign {kind} `{name}`")]
  ForeignExtension { kind: &'static str, name: String },
  #[error("Rust code generation cannot embed an external tensor in a compiled program")]
  ExternalTensor,
  #[error("Rust code generation cannot embed a runtime tensor ID in a compiled program")]
  RuntimeTensor,
  #[error("Rust code generation does not support tensor providers required by relation `{0}`")]
  TensorRelation(String),
  #[error("Rust code generation cannot embed a runtime symbol ID in a compiled program")]
  RuntimeSymbol,
}

impl ast::Program {
  pub fn to_rs_module(&self, opt: &CompileOptions) -> Result<TokenStream, RustEmissionError> {
    self.validate_rust_emission()?;
    // Precomputed cache
    let rel_to_strat_map = compute_relation_to_stratum_map(self);
    let strat_dep = compute_stratum_dependency(self, &rel_to_strat_map);
    let inters_dep = compute_interstratum_dependency(self);

    // Final output struct
    let final_result_relations = self
      .relations()
      .filter(|r| r.output == OutputOption::Default)
      .collect::<Vec<_>>();
    let final_result_fields = final_result_relations
      .iter()
      .map(|r| {
        let field = r.to_rs_output_struct_field(opt);
        quote! { pub #field }
      })
      .collect::<Vec<_>>();
    let (marker_field, marker_init) = provenance_marker(final_result_fields.is_empty());
    let final_result_struct_decl = quote! {
      #[derive(Clone)]
      pub struct OutputRelations<C: Provenance> { #(#final_result_fields)* #marker_field }
    };

    let entity_relations = self
      .adt_variant_registry
      .iter()
      .map(|(_, v)| v.relation_name.as_str())
      .collect();
    // Strata Struct/run code
    let strata_rs = self
      .strata
      .iter()
      .enumerate()
      .map(|(i, stratum)| {
        let result_struct_decl = stratum.to_rs_result_struct_decl(i, &inters_dep, opt);
        let run_fn = stratum.to_rs_run_fn(i, self, &rel_to_strat_map, &strat_dep, &inters_dep, &entity_relations);
        quote! { #result_struct_decl #run_fn }
      })
      .collect::<Vec<_>>();

    // Execute all strata
    let exec_strata = self
      .strata
      .iter()
      .enumerate()
      .map(|(i, _)| {
        let curr_strat_name = format_ident!("stratum_{}_result", i);
        let curr_strat_run_name = format_ident!("run_stratum_{}", i);
        let args = strat_dep[&i]
          .iter()
          .map(|dep_id| {
            let dep_strat_name = format_ident!("stratum_{}_result", dep_id);
            quote! { &#dep_strat_name, }
          })
          .collect::<Vec<_>>();
        quote! { let #curr_strat_name = #curr_strat_run_name(&rt, ctx, &edb, #(#args)*); }
      })
      .collect::<Vec<_>>();

    // Ensemble the final output
    let ensemble_output_relations = final_result_relations
      .iter()
      .map(|r| {
        let field_name = relation_name_to_rs_field_name(&r.predicate);
        let strat_id = rel_to_strat_map[&r.predicate];
        let strat_result_ident = format_ident!("stratum_{}_result", strat_id);
        if r.storage == StorageMetadata::IndexedVec {
          quote! { #field_name: StaticCollection::from_dynamic_elements(#strat_result_ident.#field_name)
          .recover_with_env(ctx, &rt), }
        } else if needs_externalization(&r.tuple_type) {
          quote! { #field_name: #strat_result_ident.#field_name.recover_with_env(ctx, &rt), }
        } else {
          quote! { #field_name: #strat_result_ident.#field_name.recover(ctx), }
        }
      })
      .collect::<Vec<_>>();
    let output_relations = quote! { OutputRelations { #(#ensemble_output_relations)* #marker_init } };
    let store_outputs = self.relations().filter_map(|r| {
      let OutputOption::File(OutputFile::CSV(csv)) = &r.output else {
        return None;
      };
      let file_path = csv.file_path.to_string_lossy();
      let delimiter = csv.deliminator;
      let field = r.to_rs_field_name();
      let result = format_ident!("stratum_{}_result", rel_to_strat_map[&r.predicate]);
      let tuples = if r.storage == StorageMetadata::IndexedVec {
        quote! { #result.#field.into_iter().filter_map(|e| rt.externalize_tuple(&e.tuple)) }
      } else {
        quote! { #result.#field.elements.into_iter().filter_map(|e| {
          rt.externalize_tuple(&e.tuple().into_dyn_tuple())
        }) }
      };
      Some(quote! {
        let tuples = (#tuples).collect::<Vec<_>>();
        scallop_core::runtime::database::io::store_csv_tuples(
          std::path::Path::new(#file_path), #delimiter, &tuples,
        )?;
      })
    });
    let input_relations = self.relations().filter(|r| r.populates_edb()).map(relation_to_rs_input);
    let adt_variants = variant_registry_to_rs(&self.adt_variant_registry);
    let rule_tags = self.predicate_registry.rule_tags().map(|tag| {
      let name = &tag.name;
      let ty = value_type_to_rs_value(&tag.arg_ty);
      quote! { rt.predicate_registry.register(TagPredicate::new(#name.into(), #ty)).expect("unique rule tag"); }
    });

    // Composite
    Ok(quote! {
      #[allow(unused_imports)]
      use scallop_core::common::{
        adt_variant_registry::ADTVariantRegistry, binary_op::BinaryOp, expr::Expr,
        duration::Duration, foreign_aggregate::AggregateInfo, input_file::InputFile,
        input_tag::DynamicInputTag, tuple::Tuple, tuple_access::TupleAccessor, tuple_type::TupleType,
        unary_op::UnaryOp, value::{DateTime, Value}, value_type::ValueType,
      };
      #[allow(unused_imports)]
      use scallop_core::compiler::{
        front::analyzers::tagged_rule::TagPredicate,
        ram::{Fact, Relation},
      };
      #[allow(unused_imports)]
      use scallop_core::runtime::dynamic::{DynamicIndexedVecCollection, dataflow as dynops, dataflow::aggregation::*};
      use scallop_core::runtime::error::RuntimeError;
      use scallop_core::runtime::provenance::*;
      use scallop_core::runtime::env::*;
      use scallop_core::runtime::statics::*;
      use scallop_core::runtime::database::extensional::*;
      #(#strata_rs)*
      #final_result_struct_decl
      pub fn run<C: Provenance>(ctx: &mut C) -> OutputRelations<C> {
        try_run(ctx).expect("Scallop execution failed")
      }
      pub fn run_with_edb<C: Provenance>(ctx: &mut C, edb: ExtensionalDatabase<C>) -> OutputRelations<C> {
        try_run_with_edb(ctx, edb, RuntimeEnvironmentOptions::default()).expect("Scallop execution failed")
      }
      pub fn try_run<C: Provenance>(ctx: &mut C) -> Result<OutputRelations<C>, RuntimeError> {
        try_run_with_edb(ctx, ExtensionalDatabase::new(), RuntimeEnvironmentOptions::default())
      }
      pub fn try_run_with_edb<C: Provenance>(
        ctx: &mut C, mut edb: ExtensionalDatabase<C>, options: RuntimeEnvironmentOptions,
      ) -> Result<OutputRelations<C>, RuntimeError> {
        #[allow(unused_mut)]
        let mut rt = options.build();
        #(#rule_tags)*
        rt.dynamic_entity_store.update_variant_registry(#adt_variants);
        edb.populate_relations(&rt, &[#(#input_relations),*])?;
        edb.internalize(&rt, ctx);
        #(#exec_strata)*
        #(#store_outputs)*
        Ok(#output_relations)
      }
    })
  }

  pub fn to_rs_create_edb_fn(&self) -> TokenStream {
    // Generate relation types
    let relation_types = self
      .relations()
      .map(|rel| {
        let name = rel.predicate.clone();
        let ty = tuple_type_to_rs_value(&rel.tuple_type);
        quote! { (#name.to_string(), #ty) }
      })
      .collect::<Vec<_>>();

    // Generate a `create_edb` function
    quote! {
      mod create_edb_fn {
        use scallop_core::common::tuple_type::TupleType;
        use scallop_core::common::value_type::ValueType;
        use scallop_core::runtime::database::extensional::*;
        use scallop_core::runtime::provenance::*;
        pub fn create_edb<C: Provenance>() -> ExtensionalDatabase<C> {
          ExtensionalDatabase::with_relation_types(vec![ #(#relation_types),* ].into_iter())
        }
      }
      pub use create_edb_fn::create_edb;
    }
  }

  pub fn to_rs_output(&self, result_name: &str) -> TokenStream {
    let result_ident = format_ident!("{}", result_name);
    let outputs = self
      .strata
      .iter()
      .flat_map(|stratum| {
        stratum
          .relations
          .values()
          .filter_map(|relation| match &relation.output {
            OutputOption::Default => {
              let field_name = relation_name_to_rs_field_name(&relation.predicate);
              let relation_name = relation.predicate.clone();
              Some(quote! {
                println!("{}: {}", #relation_name, #result_ident.#field_name);
              })
            }
            _ => None,
          })
      })
      .collect::<Vec<_>>();
    quote! { #(#outputs)* }
  }
}

impl ast::Stratum {
  pub fn to_rs_result_struct_decl(
    &self,
    id: usize,
    inters_dep: &InterStratumDependency,
    opt: &CompileOptions,
  ) -> TokenStream {
    let struct_name = format_ident!("Stratum{}Result", id);
    let fields = self
      .relations
      .values()
      .filter_map(|r| {
        if r.output.is_not_hidden() || inters_dep.contains(&r.predicate) {
          Some(r.to_rs_result_struct_field(opt))
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    let (marker, _) = provenance_marker(fields.is_empty());
    quote! { struct #struct_name<C: Provenance> { #(#fields)* #marker } }
  }

  pub fn to_rs_run_fn(
    &self,
    id: usize,
    program: &Program,
    rel_to_strat_map: &RelationToStratumMap,
    strat_dep: &StratumDependency,
    inters_dep: &InterStratumDependency,
    entity_relations: &HashSet<&str>,
  ) -> TokenStream {
    // Signature
    let fn_name = format_ident!("run_stratum_{}", id);
    let args = strat_dep[&id].iter().map(|dep_id| {
      let stratum_input_name = format_ident!("stratum_{}_result", dep_id);
      let stratum_input_type = format_ident!("Stratum{}Result", dep_id);
      quote! { #stratum_input_name: &#stratum_input_type<C>, }
    });
    let ret_ty = format_ident!("Stratum{}Result", id);

    // 1. Create relations
    let create_relation_stmts = self
      .relations
      .iter()
      .map(|(predicate, relation)| {
        let rs_rel_name = relation_local(predicate);
        let rs_ty = tuple_type_to_rs_type(&relation.tuple_type);
        let scheduler = if relation.immutable {
          quote! { Some(Scheduler::LFP) }
        } else if let Some(scheduler) = &relation.scheduler {
          let s = scheduler_to_rs(scheduler);
          quote! { Some(#s) }
        } else {
          quote! { None }
        };
        let goal = relation.is_goal && !relation.immutable;
        quote! {
          let #rs_rel_name = iter.create_relation_with_options::<#rs_ty>(#scheduler, #goal);
          edb.load_into_static_relation(#predicate, iter.provenance_context, &#rs_rel_name);
        }
      })
      .collect::<Vec<_>>();

    // 2. Iteration
    let updates = self
      .updates
      .iter()
      .map(|update| update.to_rs_insert(id, program, rel_to_strat_map))
      .collect::<Vec<_>>();

    let variants = self
      .relations
      .keys()
      .filter(|r| entity_relations.contains(r.as_str()))
      .collect::<Vec<_>>();
    let drain_entities = if variants.is_empty() {
      quote! {}
    } else {
      let inserts = variants.iter().map(|name| {
        let relation = relation_local(name);
        quote! {
          if let Some(facts) = entities.remove(#name) {
            #relation.insert_entity_facts(iter.provenance_context, facts);
          }
        }
      });
      quote! {
        let mut entities = rt.drain_new_entities(|name| matches!(name, #(#variants)|*));
        #(#inserts)*
      }
    };

    // 2.5. complete statements
    let complete_stmts = self
      .relations
      .values()
      .map(|r| {
        let local = relation_local(&r.predicate);
        if r.storage == StorageMetadata::IndexedVec {
          quote! { let #local = iter.complete(&#local).into_indexed_vec(iter.provenance_context); }
        } else {
          quote! { let #local = iter.complete(&#local); }
        }
      })
      .collect::<Vec<_>>();
    let complete_stmts = quote! { #(#complete_stmts)* };

    // 3. Ensemble final result
    let ensemble_result_fields = self
      .relations
      .values()
      .filter_map(|r| {
        if r.output.is_not_hidden() || inters_dep.contains(&r.predicate) {
          let rs_name = relation_name_to_rs_field_name(&r.predicate);
          let local = relation_local(&r.predicate);
          Some(quote! { #rs_name: #local, })
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    let (_, marker) = provenance_marker(ensemble_result_fields.is_empty());
    let ensemble_result = quote! { #ret_ty { #(#ensemble_result_fields)* #marker } };

    // Final function
    quote! {
      fn #fn_name<C: Provenance>(rt: &RuntimeEnvironment, ctx: &mut C, edb: &ExtensionalDatabase<C>, #(#args)*) -> #ret_ty<C> {
        let mut iter = StaticIteration::<C>::new(rt, ctx);
        #(#create_relation_stmts)*
        while iter.need_to_iterate() {
          #(#updates)*
          #drain_entities
          iter.step();
        }
        #complete_stmts
        #ensemble_result
      }
    }
  }
}

impl ast::Relation {
  pub fn to_rs_field_name(&self) -> TokenStream {
    relation_name_to_rs_field_name(&self.predicate)
  }

  pub fn to_rs_result_struct_field(&self, _: &CompileOptions) -> TokenStream {
    let field_name = relation_name_to_rs_field_name(&self.predicate);
    if self.storage == StorageMetadata::IndexedVec {
      quote! { #field_name: DynamicIndexedVecCollection<C>, }
    } else {
      let ty = tuple_type_to_rs_type(&self.tuple_type);
      quote! { #field_name: StaticCollection<#ty, C>, }
    }
  }

  pub fn to_rs_output_struct_field(&self, _: &CompileOptions) -> TokenStream {
    let field_name = relation_name_to_rs_field_name(&self.predicate);
    let ty = tuple_type_to_rs_type(&self.tuple_type);
    quote! { #field_name: StaticOutputCollection<#ty, C>, }
  }
}

impl ast::Update {
  fn to_rs_insert(
    &self,
    curr_strat_id: usize,
    program: &Program,
    rel_to_strat_map: &RelationToStratumMap,
  ) -> TokenStream {
    let rs_rel_name = relation_local(&self.target);
    let shared = needs_shared_ops(&self.dataflow)
      || self.dependency().iter().chain([&self.target]).any(|name| {
        let r = &program.strata[rel_to_strat_map[name]].relations[name];
        is_wide(&r.tuple_type) || r.storage == StorageMetadata::IndexedVec
      });
    let context = EmissionContext {
      shared,
      program,
      curr_strat_id,
      rel_to_strat_map,
    };
    let rs_dataflow = context.emit_dataflow(&self.dataflow);
    let rs_dataflow = if shared {
      quote! { dataflow::from_dynamic(#rs_dataflow) }
    } else {
      rs_dataflow
    };
    quote! { iter.insert_dataflow(&#rs_rel_name, #rs_dataflow); }
  }
}

struct EmissionContext<'a> {
  shared: bool,
  program: &'a Program,
  curr_strat_id: usize,
  rel_to_strat_map: &'a RelationToStratumMap,
}

impl EmissionContext<'_> {
  fn emit_relation(&self, name: &str) -> TokenStream {
    let stratum_id = self.rel_to_strat_map[name];
    if stratum_id != self.curr_strat_id
      && self.program.strata[stratum_id].relations[name].storage == StorageMetadata::IndexedVec
    {
      let collection = self.previous_collection(name);
      return quote! {
        dynops::DynamicDataflow::dynamic_collection(
          scallop_core::runtime::dynamic::DynamicCollectionRef::IndexedVec(#collection),
          iter.is_first_iteration(),
        )
      };
    }
    let relation = if stratum_id == self.curr_strat_id {
      let local = relation_local(name);
      quote! { &#local }
    } else {
      let collection = self.previous_collection(name);
      quote! { dataflow::collection(#collection, iter.is_first_iteration()) }
    };
    if self.shared {
      quote! { dataflow::to_dynamic(#relation) }
    } else {
      relation
    }
  }

  fn previous_collection(&self, name: &str) -> TokenStream {
    let field = relation_name_to_rs_field_name(name);
    let stratum = format_ident!("stratum_{}_result", self.rel_to_strat_map[name]);
    quote! { &#stratum.#field }
  }

  fn emit_dataflow(&self, flow: &Dataflow) -> TokenStream {
    use Dataflow::*;
    let ctx = quote! { iter.provenance_context };
    let emit = |child| self.emit_dataflow(child);
    match flow {
      Unit(ty) if self.shared => {
        let ty = tuple_type_to_rs_value(ty);
        quote! { if iter.is_first_iteration() {
          dynops::DynamicDataflow::recent_unit(#ctx, #ty)
        } else {
          dynops::DynamicDataflow::stable_unit(#ctx, #ty)
        } }
      }
      Unit(ty) => {
        let ty = tuple_type_to_rs_type(ty);
        quote! { iter.unit::<#ty>(iter.is_first_iteration()) }
      }
      UntaggedVec(tuples) => {
        let tuples = tuples.iter().map(tuple_to_rs_value);
        quote! { dynops::DynamicDataflow::untagged_vec(#ctx, vec![
          #(rt.internalize_tuple(&#tuples).expect("internalize program tuple"),)*
        ]) }
      }
      Relation(name) => self.emit_relation(name),
      Project(d, expr) | Filter(d, expr) => {
        let input = emit(d);
        let method = if matches!(flow, Project(..)) {
          format_ident!("project")
        } else {
          format_ident!("filter")
        };
        if self.shared {
          let expr = expr_to_rs_value(expr);
          quote! { (#input).#method(rt.internalize_expr(&#expr).expect("internalize program expression"), rt) }
        } else {
          let expr = structural_expr_to_rs(expr);
          quote! { dataflow::#method(#input, |t| #expr) }
        }
      }
      Find(d, key) => {
        let input = emit(d);
        if self.shared {
          let key = tuple_to_rs_value(key);
          quote! { (#input).find(rt.internalize_tuple(&#key).expect("internalize program key")) }
        } else {
          let key = tuple_to_rs_tuple(key);
          quote! { dataflow::find(#input, #key) }
        }
      }
      Sorted(d) => {
        let input = emit(d);
        if self.shared {
          quote! { (#input).sorted() }
        } else {
          quote! { dataflow::sorted(#input) }
        }
      }
      Union(l, r) | Join(l, r) | Intersect(l, r) | Product(l, r) | Antijoin(l, r) | Difference(l, r) => {
        let left = emit(l);
        let right = emit(r);
        let op = match flow {
          Union(..) => "union",
          Join(..) => "join",
          Intersect(..) => "intersect",
          Product(..) => "product",
          Antijoin(..) => "antijoin",
          Difference(..) => "difference",
          _ => unreachable!(),
        };
        let method = format_ident!("{op}");
        if !self.shared {
          quote! { iter.#method(#left, #right) }
        } else if matches!(flow, Union(..)) {
          quote! { (#left).union(#right) }
        } else {
          quote! { (#left).#method(#right, #ctx) }
        }
      }
      JoinIndexedVec(left, name) => {
        let input = emit(left);
        let index = self.previous_collection(name);
        quote! { (#input).join_indexed_vec(#index, #ctx) }
      }
      Reduce(r) => {
        let input = self.emit_relation(&r.predicate);
        let name = &r.aggregator;
        let info = aggregate_info_to_rs(&r.aggregate_info);
        let aggregate = quote! {
          rt.aggregate_registry.instantiate_aggregator::<C>(#name, #info).expect("built-in aggregate")
        };
        let reduce = match &r.group_by {
          ReduceGroupByType::None => {
            quote! { DynamicAggregationSingleGroupDataflow::new(#aggregate, #input, #ctx, rt) }
          }
          ReduceGroupByType::Implicit => {
            quote! { DynamicAggregationImplicitGroupDataflow::new(#aggregate, #input, #ctx, rt) }
          }
          ReduceGroupByType::Join(group) => {
            let group = self.emit_relation(group);
            quote! { DynamicAggregationJoinGroupDataflow::new(#aggregate, #group, #input, #ctx, rt) }
          }
        };
        quote! { dynops::DynamicDataflow::new(#reduce) }
      }
      OverwriteOne(d) => {
        let input = emit(d);
        quote! { (#input).overwrite_one(#ctx) }
      }
      Exclusion(l, r) => {
        let left = emit(l);
        let right = emit(r);
        quote! { (#left).dynamic_exclusion(#right, #ctx, rt) }
      }
      ForeignPredicateGround(name, args) => {
        let args = args.iter().map(value_to_rs_value);
        quote! { dynops::DynamicDataflow::foreign_predicate_ground(
          #name.into(), vec![#(rt.internalize_value(&#args).expect("internalize built-in predicate argument")),*],
          iter.is_first_iteration(), #ctx, rt,
        ) }
      }
      ForeignPredicateConstraint(d, name, args) | ForeignPredicateJoin(d, name, args) => {
        let input = emit(d);
        let args = args.iter().map(expr_to_rs_value);
        let method = if matches!(flow, ForeignPredicateConstraint(..)) {
          format_ident!("foreign_predicate_constraint")
        } else {
          format_ident!("foreign_predicate_join")
        };
        quote! { (#input).#method(#name.into(), vec![#(#args),*], #ctx, rt) }
      }
    }
  }
}

fn relation_name_to_rs_field_name(n: &str) -> TokenStream {
  let name = if n.contains("#") {
    format!(
      "_{}",
      n.replace("#", "_p")
        .replace("(", "_l")
        .replace(")", "_r")
        .replace(",", "_c")
    )
  } else {
    n.to_string()
  };
  let ident = match name.as_str() {
    "self" | "Self" | "super" | "crate" | "_" => format_ident!("_rust_{name}"),
    name => Ident::new_raw(name, Span::call_site()),
  };
  quote! { #ident }
}

fn relation_local(name: &str) -> TokenStream {
  relation_name_to_rs_field_name(&format!("relation_{name}"))
}

fn provenance_marker(empty: bool) -> (TokenStream, TokenStream) {
  if empty {
    (
      quote! { _provenance: std::marker::PhantomData<C>, },
      quote! { _provenance: std::marker::PhantomData, },
    )
  } else {
    Default::default()
  }
}

// Standard Rust tuples implement PartialOrd through arity twelve.
const TYPED_TUPLE_LIMIT: usize = 12;

fn tuple_type_to_rs_type(ty: &TupleType) -> TokenStream {
  match ty {
    TupleType::Tuple(t) if t.len() > TYPED_TUPLE_LIMIT => quote! { Tuple },
    TupleType::Tuple(t) if t.is_empty() => quote! { () },
    TupleType::Tuple(t) => {
      let elems = t.iter().map(tuple_type_to_rs_type).collect::<Vec<_>>();
      quote! { (#(#elems),*,) }
    }
    TupleType::Value(v) => value_type_to_rs_type(v),
  }
}

fn value_type_to_rs_type(ty: &ValueType) -> TokenStream {
  match ty {
    ValueType::I8 => quote! { i8 },
    ValueType::I16 => quote! { i16 },
    ValueType::I32 => quote! { i32 },
    ValueType::I64 => quote! { i64 },
    ValueType::I128 => quote! { i128 },
    ValueType::ISize => quote! { isize },
    ValueType::U8 => quote! { u8 },
    ValueType::U16 => quote! { u16 },
    ValueType::U32 => quote! { u32 },
    ValueType::U64 => quote! { u64 },
    ValueType::U128 => quote! { u128 },
    ValueType::USize => quote! { usize },
    ValueType::F32 => quote! { f32 },
    ValueType::F64 => quote! { f64 },
    ValueType::Bool => quote! { bool },
    ValueType::Char => quote! { char },
    ValueType::Str => quote! { &'static str },
    ValueType::String => quote! { String },
    ValueType::DateTime => quote! { DateTime },
    ValueType::Duration => quote! { Duration },
    ValueType::Symbol | ValueType::Entity | ValueType::Tensor => quote! { Value },
  }
}

fn structural_expr_to_rs(expr: &Expr) -> TokenStream {
  match expr {
    Expr::Tuple(t) => {
      if t.is_empty() {
        quote! { () }
      } else {
        let elems = t.iter().map(structural_expr_to_rs).collect::<Vec<_>>();
        quote! { (#(#elems),*,) }
      }
    }
    Expr::Access(a) => {
      if a.is_empty() {
        quote! { t.clone() }
      } else {
        let indices = a.iter().map(Literal::usize_unsuffixed);
        quote! { t.#(#indices).*.clone() }
      }
    }
    Expr::Constant(c) => value_to_rs_literal(c),
    _ => unreachable!("non-structural expressions use shared operators"),
  }
}

fn expr_to_rs_value(expr: &Expr) -> TokenStream {
  match expr {
    Expr::Tuple(t) => {
      let elems = t.iter().map(expr_to_rs_value);
      quote! { Expr::Tuple(vec![#(#elems),*]) }
    }
    Expr::Access(a) => {
      let len = a.len;
      let indices = a.indices;
      quote! { Expr::Access(TupleAccessor { len: #len, indices: [#(#indices),*] }) }
    }
    Expr::Constant(c) => {
      let value = value_to_rs_value(c);
      quote! { Expr::Constant(#value) }
    }
    Expr::Binary(b) => {
      let op1 = expr_to_rs_value(&b.op1);
      let op2 = expr_to_rs_value(&b.op2);
      let op = variant_ident(&b.op);
      quote! { Expr::binary(BinaryOp::#op, #op1, #op2) }
    }
    Expr::Unary(u) => {
      let op1 = expr_to_rs_value(&u.op1);
      let op = match &u.op {
        UnaryOp::TypeCast(target_ty) => {
          let ty = value_type_to_rs_value(target_ty);
          quote! { UnaryOp::TypeCast(#ty) }
        }
        op => {
          let op = variant_ident(op);
          quote! { UnaryOp::#op }
        }
      };
      quote! { Expr::unary(#op, #op1) }
    }
    Expr::IfThenElse(i) => {
      let cond = expr_to_rs_value(&i.cond);
      let then_br = expr_to_rs_value(&i.then_br);
      let else_br = expr_to_rs_value(&i.else_br);
      quote! { Expr::ite(#cond, #then_br, #else_br) }
    }
    Expr::Call(c) => {
      let function = &c.function;
      let args = c.args.iter().map(expr_to_rs_value);
      quote! { Expr::call(#function.into(), vec![#(#args),*]) }
    }
    Expr::New(n) => {
      let functor = &n.functor;
      let args = n.args.iter().map(expr_to_rs_value);
      quote! { Expr::new(#functor.into(), vec![#(#args),*]) }
    }
  }
}

fn input_tag_to_rs_input_tag(tag: &DynamicInputTag) -> TokenStream {
  match tag {
    DynamicInputTag::None => quote! { DynamicInputTag::None },
    DynamicInputTag::NewVariable => quote! { DynamicInputTag::NewVariable },
    DynamicInputTag::Exclusive(i) => quote! { DynamicInputTag::Exclusive(#i) },
    DynamicInputTag::Bool(b) => quote! { DynamicInputTag::Bool(#b) },
    DynamicInputTag::Natural(n) => quote! { DynamicInputTag::Natural(#n) },
    DynamicInputTag::Float(f) => {
      let f = f.to_bits();
      quote! { DynamicInputTag::Float(f64::from_bits(#f)) }
    }
    DynamicInputTag::ExclusiveFloat(f, u) => {
      let f = f.to_bits();
      quote! { DynamicInputTag::ExclusiveFloat(f64::from_bits(#f), #u) }
    }
    DynamicInputTag::FloatWithID(id, f) => {
      let f = f.to_bits();
      quote! { DynamicInputTag::FloatWithID(#id, f64::from_bits(#f)) }
    }
    DynamicInputTag::ExclusiveFloatWithID(id, f, u) => {
      let f = f.to_bits();
      quote! { DynamicInputTag::ExclusiveFloatWithID(#id, f64::from_bits(#f), #u) }
    }
    DynamicInputTag::Tensor(_) => unreachable!("external constants are validated before emission"),
  }
}

fn tuple_to_rs_tuple(tuple: &Tuple) -> TokenStream {
  match tuple {
    Tuple::Value(v) => value_to_rs_literal(v),
    Tuple::Tuple(t) if t.is_empty() => quote! { () },
    Tuple::Tuple(t) => {
      let elems = t.iter().map(tuple_to_rs_tuple).collect::<Vec<_>>();
      quote! { (#(#elems),*,) }
    }
  }
}

fn tuple_to_rs_value(tuple: &Tuple) -> TokenStream {
  match tuple {
    Tuple::Tuple(t) => {
      let elems = t.iter().map(tuple_to_rs_value);
      quote! { Tuple::Tuple(Box::new([#(#elems),*])) }
    }
    Tuple::Value(v) => {
      let value = value_to_rs_value(v);
      quote! { Tuple::Value(#value) }
    }
  }
}

fn value_to_rs_literal(value: &Value) -> TokenStream {
  use Value::*;
  match value {
    I8(i) => quote! { #i },
    I16(i) => quote! { #i },
    I32(i) => quote! { #i },
    I64(i) => quote! { #i },
    I128(i) => quote! { #i },
    ISize(i) => quote! { #i },
    U8(u) => quote! { #u },
    U16(u) => quote! { #u },
    U32(u) => quote! { #u },
    U64(u) => quote! { #u },
    U128(u) => quote! { #u },
    USize(u) => quote! { #u },
    F32(f) => {
      let bits = f.to_bits();
      quote! { f32::from_bits(#bits) }
    }
    F64(f) => {
      let bits = f.to_bits();
      quote! { f64::from_bits(#bits) }
    }
    Char(c) => quote! { #c },
    Bool(b) => quote! { #b },
    Str(s) => quote! { #s },
    String(s) => quote! { String::from(#s) },
    SymbolString(s) | EntityString(s) => quote! { String::from(#s) },
    DateTime(t) => {
      let secs = t.as_second();
      let nanos = t.subsec_nanosecond();
      quote! { DateTime::new(#secs, #nanos).expect("valid program timestamp") }
    }
    Duration(d) => {
      let (months, elapsed) = d.components();
      let elapsed = elapsed.to_string();
      quote! { Duration::new(#months, #elapsed.parse().expect("valid program duration")) }
    }
    Entity(e) => quote! { #e },
    Symbol(_) | Tensor(_) | TensorValue(_) => unreachable!("external constants are validated before emission"),
  }
}

fn value_to_rs_value(value: &Value) -> TokenStream {
  let variant = match value {
    Value::SymbolString(_) => format_ident!("SymbolString"),
    Value::EntityString(_) => format_ident!("EntityString"),
    _ => variant_ident(&value.value_type()),
  };
  let literal = value_to_rs_literal(value);
  quote! { Value::#variant(#literal) }
}

fn value_type_to_rs_value(ty: &ValueType) -> TokenStream {
  let name = variant_ident(ty);
  quote! { ValueType::#name }
}

fn tuple_type_to_rs_value(ty: &TupleType) -> TokenStream {
  match ty {
    TupleType::Tuple(t) => {
      let elems = t.iter().map(tuple_type_to_rs_value);
      quote! { TupleType::Tuple(Box::new([#(#elems),*])) }
    }
    TupleType::Value(v) => {
      let ty = value_type_to_rs_value(v);
      quote! { TupleType::Value(#ty) }
    }
  }
}

fn aggregate_info_to_rs(info: &AggregateInfo) -> TokenStream {
  let exclamation = info.has_exclamation_mark;
  let positional = info.pos_params.iter().map(value_to_rs_value);
  let named = info.named_params.iter().map(|(key, value)| {
    let value = value_to_rs_value(value);
    quote! { (#key.into(), #value) }
  });
  let args = info.arg_var_types.iter().map(value_type_to_rs_value);
  let inputs = info.input_var_types.iter().map(value_type_to_rs_value);
  quote! { AggregateInfo {
    pos_params: vec![#(#positional),*],
    named_params: [#(#named),*].into_iter().collect(),
    has_exclamation_mark: #exclamation,
    arg_var_types: vec![#(#args),*],
    input_var_types: vec![#(#inputs),*],
  } }
}

fn variant_registry_to_rs(registry: &ADTVariantRegistry) -> TokenStream {
  let variants = registry.iter().map(|(name, variant)| {
    let relation = &variant.relation_name;
    let args = variant.arg_types.iter().map(value_type_to_rs_value);
    quote! { variants.add(#name.into(), #relation.into(), vec![#(#args),*]); }
  });
  quote! {{
    #[allow(unused_mut)]
    let mut variants = ADTVariantRegistry::new();
    #(#variants)*
    variants
  }}
}

fn scheduler_to_rs(scheduler: &Scheduler) -> TokenStream {
  match scheduler {
    Scheduler::Beam { beam_size } => quote! { Scheduler::Beam { beam_size: #beam_size } },
    scheduler => {
      let name = variant_ident(scheduler);
      quote! { Scheduler::#name }
    }
  }
}

fn relation_to_rs_input(relation: &Relation) -> TokenStream {
  let name = &relation.predicate;
  let ty = tuple_type_to_rs_value(&relation.tuple_type);
  let storage = variant_ident(&relation.storage);
  let facts = relation.facts.iter().map(|f| {
    let tag = input_tag_to_rs_input_tag(&f.tag);
    let tuple = tuple_to_rs_value(&f.tuple);
    quote! { Fact { tag: #tag, tuple: #tuple } }
  });
  let input_file = match &relation.input_file {
    Some(InputFile::Csv {
      file_path,
      deliminator,
      has_header,
      has_probability,
      keys,
      fields,
    }) => {
      let file_path = file_path.to_string_lossy();
      let strings = |v: &Option<Vec<String>>| match v {
        Some(v) => quote! { Some(vec![#(#v.into()),*]) },
        None => quote! { None },
      };
      let keys = strings(keys);
      let fields = strings(fields);
      quote! { Some(InputFile::Csv {
        file_path: #file_path.into(), deliminator: #deliminator,
        has_header: #has_header, has_probability: #has_probability, keys: #keys, fields: #fields,
      }) }
    }
    None => quote! { None },
  };
  quote! { Relation {
    input_file: #input_file,
    facts: vec![#(#facts),*],
    storage: scallop_core::runtime::database::StorageMetadata::#storage,
    ..Relation::hidden_relation(#name.into(), #ty)
  } }
}

fn variant_ident(variant: &impl std::fmt::Debug) -> Ident {
  Ident::new(&format!("{variant:?}"), Span::call_site())
}

fn is_wide(ty: &TupleType) -> bool {
  matches!(ty, TupleType::Tuple(t) if t.len() > TYPED_TUPLE_LIMIT || t.iter().any(is_wide))
}

fn contains_tensor(ty: &TupleType) -> bool {
  match ty {
    TupleType::Tuple(t) => t.iter().any(contains_tensor),
    TupleType::Value(v) => *v == ValueType::Tensor,
  }
}

fn needs_externalization(ty: &TupleType) -> bool {
  match ty {
    TupleType::Tuple(t) => t.iter().any(needs_externalization),
    TupleType::Value(v) => matches!(v, ValueType::Symbol | ValueType::Tensor),
  }
}

fn needs_internalization(value: &Value) -> bool {
  matches!(
    value,
    Value::SymbolString(_) | Value::Entity(_) | Value::EntityString(_)
  )
}

fn is_structural(expr: &Expr) -> bool {
  match expr {
    Expr::Tuple(t) => t.len() <= TYPED_TUPLE_LIMIT && t.iter().all(is_structural),
    Expr::Access(_) => true,
    Expr::Constant(v) => !needs_internalization(v),
    _ => false,
  }
}

fn needs_shared_ops(flow: &Dataflow) -> bool {
  use Dataflow::*;
  let shared = match flow {
    Unit(ty) => !matches!(ty, TupleType::Tuple(t) if t.is_empty()),
    Project(_, e) | Filter(_, e) => !is_structural(e),
    Find(_, key) => is_wide(&key.tuple_type()) || tuple_needs_internalization(key),
    UntaggedVec(_)
    | JoinIndexedVec(..)
    | Reduce(_)
    | OverwriteOne(_)
    | Exclusion(..)
    | ForeignPredicateGround(..)
    | ForeignPredicateConstraint(..)
    | ForeignPredicateJoin(..) => true,
    _ => false,
  };
  shared || flow.children().any(needs_shared_ops)
}

impl Program {
  fn validate_rust_emission(&self) -> Result<(), RustEmissionError> {
    Self::validate_rust_extensions(
      &self.function_registry,
      &self.predicate_registry,
      &self.aggregate_registry,
    )?;
    for relation in self.relations() {
      if contains_tensor(&relation.tuple_type) {
        return Err(RustEmissionError::TensorRelation(relation.predicate.clone()));
      }
      for fact in &relation.facts {
        if matches!(fact.tag, DynamicInputTag::Tensor(_)) {
          return Err(RustEmissionError::ExternalTensor);
        }
        validate_tuple(&fact.tuple)?;
      }
    }
    for update in self.strata.iter().flat_map(|s| &s.updates) {
      self.validate_flow(&update.dataflow)?;
    }
    Ok(())
  }

  pub(crate) fn validate_rust_extensions(
    functions: &ForeignFunctionRegistry,
    predicates: &ForeignPredicateRegistry,
    aggregates: &AggregateRegistry,
  ) -> Result<(), RustEmissionError> {
    let extensions = functions
      .into_iter()
      .filter(|(name, _)| !functions.is_builtin(name))
      .map(|(name, _)| ("function", name))
      .chain(
        predicates
          .iter()
          .filter(|(name, _)| !predicates.is_builtin(name))
          .map(|(name, _)| ("predicate", name)),
      )
      .chain(
        aggregates
          .iter()
          .filter(|(name, _)| !aggregates.is_builtin(name))
          .map(|(name, _)| ("aggregate", name)),
      );
    if let Some((kind, name)) = extensions.min() {
      return Err(RustEmissionError::ForeignExtension {
        kind,
        name: name.clone(),
      });
    }
    Ok(())
  }

  fn validate_flow(&self, flow: &Dataflow) -> Result<(), RustEmissionError> {
    use Dataflow::*;
    match flow {
      UntaggedVec(tuples) => tuples.iter().try_for_each(validate_tuple)?,
      Project(_, e) | Filter(_, e) => self.validate_expr(e)?,
      Find(_, key) => validate_tuple(key)?,
      Reduce(r) => {
        check_builtin(
          "aggregate",
          &r.aggregator,
          self.aggregate_registry.is_builtin(&r.aggregator),
        )?;
        r.aggregate_info
          .pos_params
          .iter()
          .chain(r.aggregate_info.named_params.values())
          .try_for_each(validate_value)?;
      }
      ForeignPredicateGround(name, args) => {
        check_builtin("predicate", name, self.predicate_registry.is_builtin(name))?;
        args.iter().try_for_each(validate_value)?;
      }
      ForeignPredicateConstraint(_, name, args) | ForeignPredicateJoin(_, name, args) => {
        check_builtin("predicate", name, self.predicate_registry.is_builtin(name))?;
        args.iter().try_for_each(|expr| self.validate_expr(expr))?;
      }
      _ => {}
    }
    flow.children().try_for_each(|child| self.validate_flow(child))
  }

  fn validate_expr(&self, expr: &Expr) -> Result<(), RustEmissionError> {
    match expr {
      Expr::Tuple(t) => t.iter().try_for_each(|e| self.validate_expr(e)),
      Expr::Constant(v) => validate_value(v),
      Expr::Binary(b) => [b.op1.as_ref(), b.op2.as_ref()]
        .into_iter()
        .try_for_each(|e| self.validate_expr(e)),
      Expr::Unary(u) => self.validate_expr(&u.op1),
      Expr::IfThenElse(i) => [i.cond.as_ref(), i.then_br.as_ref(), i.else_br.as_ref()]
        .into_iter()
        .try_for_each(|e| self.validate_expr(e)),
      Expr::Call(c) => {
        check_builtin("function", &c.function, self.function_registry.is_builtin(&c.function))?;
        c.args.iter().try_for_each(|e| self.validate_expr(e))
      }
      Expr::New(n) => n.args.iter().try_for_each(|e| self.validate_expr(e)),
      Expr::Access(_) => Ok(()),
    }
  }
}

fn check_builtin(kind: &'static str, name: &str, builtin: bool) -> Result<(), RustEmissionError> {
  if builtin {
    Ok(())
  } else {
    Err(RustEmissionError::ForeignExtension {
      kind,
      name: name.into(),
    })
  }
}

fn validate_value(value: &Value) -> Result<(), RustEmissionError> {
  match value {
    Value::Tensor(_) => Err(RustEmissionError::ExternalTensor),
    Value::TensorValue(_) => Err(RustEmissionError::RuntimeTensor),
    Value::Symbol(_) => Err(RustEmissionError::RuntimeSymbol),
    _ => Ok(()),
  }
}

fn validate_tuple(tuple: &Tuple) -> Result<(), RustEmissionError> {
  match tuple {
    Tuple::Tuple(t) => t.iter().try_for_each(validate_tuple),
    Tuple::Value(v) => validate_value(v),
  }
}

fn tuple_needs_internalization(tuple: &Tuple) -> bool {
  match tuple {
    Tuple::Tuple(t) => t.iter().any(tuple_needs_internalization),
    Tuple::Value(v) => needs_internalization(v),
  }
}

type RelationToStratumMap = HashMap<String, usize>;

fn compute_relation_to_stratum_map(ast: &ast::Program) -> RelationToStratumMap {
  let mut map = RelationToStratumMap::new();
  for (i, stratum) in ast.strata.iter().enumerate() {
    for predicate in stratum.relations.keys() {
      map.insert(predicate.clone(), i);
    }
  }
  map
}

type StratumDependency = HashMap<usize, BTreeSet<usize>>;

fn compute_stratum_dependency(ast: &ast::Program, rel_to_strat_map: &RelationToStratumMap) -> StratumDependency {
  let mut dep = StratumDependency::new();
  for (i, stratum) in ast.strata.iter().enumerate() {
    let dep_rels = stratum.dependency();
    let dep_strats = dep_rels.into_iter().map(|r| rel_to_strat_map[&r]).collect();
    dep.insert(i, dep_strats);
  }
  dep
}

type InterStratumDependency = HashSet<String>;

fn compute_interstratum_dependency(ast: &ast::Program) -> InterStratumDependency {
  let mut dep = HashSet::new();
  for stratum in &ast.strata {
    let dep_rels = stratum.dependency();
    dep.extend(dep_rels.into_iter().filter(|r| !stratum.relations.contains_key(r)));
  }
  dep
}
