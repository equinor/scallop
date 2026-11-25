use super::*;

use crate::runtime::env::*;
use crate::runtime::provenance::*;

pub struct StaticIteration<'a, Prov: Provenance> {
  pub iter_num: usize,
  pub early_discard: bool,
  relations: Vec<(Box<dyn StaticRelationTrait<Prov>>, Scheduler, bool)>,
  pub runtime_environment: &'a RuntimeEnvironment,
  pub provenance_context: &'a mut Prov,
}

impl<'a, Prov: Provenance> StaticIteration<'a, Prov> {
  /// Create a new Iteration
  pub fn new(runtime_environment: &'a RuntimeEnvironment, provenance_context: &'a mut Prov) -> Self {
    Self {
      iter_num: 0,
      early_discard: runtime_environment.early_discard,
      runtime_environment,
      provenance_context,
      relations: Vec::new(),
    }
  }

  pub fn is_first_iteration(&self) -> bool {
    self.iter_num == 0
  }

  pub fn step(&mut self) {
    self.iter_num += 1;
  }

  pub fn changed(&mut self) -> bool {
    let mut result = false;
    for (relation, scheduler, _) in &mut self.relations {
      if relation.changed_with_scheduler(self.provenance_context, scheduler) {
        result = true;
      }
    }
    result
  }

  pub fn need_to_iterate(&mut self) -> bool {
    let criteria = &self.runtime_environment.stopping_criteria;
    let limit = criteria.get_iter_limit();
    if self.changed() || self.is_first_iteration() {
      let mut goals = self.relations.iter().filter(|(_, _, goal)| *goal).peekable();
      let reached_goal = criteria.stop_when_goal_relation_non_empty()
        && goals.peek().is_some()
        && goals.all(|(relation, _, _)| relation.has_recent());
      if limit.is_some_and(|limit| self.iter_num > limit) || reached_goal {
        for (relation, _, _) in &self.relations {
          relation.finish_early(self.provenance_context);
        }
        return false;
      }
      true
    } else {
      limit.is_some_and(|limit| self.iter_num < limit)
    }
  }

  pub fn create_relation<Tup>(&mut self) -> StaticRelation<Tup, Prov>
  where
    Tup: StaticTupleTrait + 'static,
  {
    self.create_relation_with_options(None, false)
  }

  pub fn create_relation_with_options<Tup: StaticTupleTrait>(
    &mut self,
    scheduler: Option<Scheduler>,
    is_goal: bool,
  ) -> StaticRelation<Tup, Prov> {
    let relation = StaticRelation::new();
    let scheduler = scheduler.unwrap_or_else(|| self.runtime_environment.get_default_scheduler().clone());
    self.relations.push((Box::new(relation.clone()), scheduler, is_goal));
    relation
  }

  pub fn insert_dataflow<D, Tup>(&self, r: &StaticRelation<Tup, Prov>, data: D)
  where
    D: dataflow::Dataflow<Tup, Prov>,
    Tup: StaticTupleTrait,
  {
    r.insert_dataflow_recent(self.provenance_context, data, self.early_discard)
  }

  pub fn unit<U: dataflow::UnitTuple>(&self, first_iteration: bool) -> dataflow::Unit<'_, U, Prov> {
    dataflow::unit::<U, Prov>(self.provenance_context, first_iteration)
  }

  pub fn product<D1, D2, T1, T2>(&self, v1: D1, v2: D2) -> dataflow::Product<'_, D1, D2, T1, T2, Prov>
  where
    T1: StaticTupleTrait,
    T2: StaticTupleTrait,
    D1: dataflow::Dataflow<T1, Prov>,
    D2: dataflow::Dataflow<T2, Prov>,
  {
    dataflow::product(v1, v2, self.provenance_context)
  }

  pub fn intersect<D1, D2, Tup>(&self, v1: D1, v2: D2) -> dataflow::Intersection<'_, D1, D2, Tup, Prov>
  where
    Tup: StaticTupleTrait,
    D1: dataflow::Dataflow<Tup, Prov>,
    D2: dataflow::Dataflow<Tup, Prov>,
  {
    dataflow::intersect(v1, v2, self.provenance_context)
  }

  pub fn union<D1, D2, Tup>(&self, v1: D1, v2: D2) -> dataflow::Union<'_, D1, D2, Tup, Prov>
  where
    Tup: StaticTupleTrait,
    D1: dataflow::Dataflow<Tup, Prov>,
    D2: dataflow::Dataflow<Tup, Prov>,
  {
    dataflow::union(v1, v2, self.provenance_context)
  }

  pub fn join<D1, D2, K, T1, T2>(&self, v1: D1, v2: D2) -> dataflow::Join<'_, D1, D2, K, T1, T2, Prov>
  where
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    T2: StaticTupleTrait,
    D1: dataflow::Dataflow<(K, T1), Prov>,
    D2: dataflow::Dataflow<(K, T2), Prov>,
  {
    dataflow::join(v1, v2, self.provenance_context)
  }

  pub fn difference<D1, D2, Tup>(&self, v1: D1, v2: D2) -> dataflow::Difference<'_, D1, D2, Tup, Prov>
  where
    Tup: StaticTupleTrait,
    D1: dataflow::Dataflow<Tup, Prov>,
    D2: dataflow::Dataflow<Tup, Prov>,
  {
    dataflow::difference(v1, v2, self.provenance_context)
  }

  pub fn antijoin<D1, D2, K, T1>(&self, v1: D1, v2: D2) -> dataflow::Antijoin<'_, D1, D2, K, T1, Prov>
  where
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    D1: dataflow::Dataflow<(K, T1), Prov>,
    D2: dataflow::Dataflow<K, Prov>,
  {
    dataflow::antijoin(v1, v2, self.provenance_context)
  }

  pub fn aggregate<A, D, T1>(&self, agg: A, d: D) -> dataflow::AggregationSingleGroup<'_, A, D, T1, Prov>
  where
    A: Aggregator<T1, Prov>,
    T1: StaticTupleTrait,
    D: dataflow::Dataflow<T1, Prov>,
  {
    dataflow::AggregationSingleGroup::new(agg, d, self.runtime_environment, self.provenance_context)
  }

  pub fn aggregate_implicit_group<A, D, K, T1>(
    &self,
    agg: A,
    d: D,
  ) -> dataflow::AggregationImplicitGroup<'_, A, D, K, T1, Prov>
  where
    A: Aggregator<T1, Prov>,
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    D: dataflow::Dataflow<(K, T1), Prov>,
  {
    dataflow::AggregationImplicitGroup::new(agg, d, self.runtime_environment, self.provenance_context)
  }

  pub fn aggregate_join_group<A, D1, D2, K, T1, T2>(
    &self,
    agg: A,
    v1: D1,
    v2: D2,
  ) -> dataflow::AggregationJoinGroup<'_, A, D1, D2, K, T1, T2, Prov>
  where
    A: Aggregator<T2, Prov>,
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    T2: StaticTupleTrait,
    D1: dataflow::Dataflow<(K, T1), Prov>,
    D2: dataflow::Dataflow<(K, T2), Prov>,
  {
    dataflow::AggregationJoinGroup::new(agg, v1, v2, self.runtime_environment, self.provenance_context)
  }

  pub fn complete<Tup>(&self, r: &StaticRelation<Tup, Prov>) -> StaticCollection<Tup, Prov>
  where
    Tup: StaticTupleTrait,
  {
    r.complete(self.provenance_context)
  }
}
