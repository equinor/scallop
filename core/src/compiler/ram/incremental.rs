use petgraph::visit::{Dfs, VisitMap, Visitable};
use std::collections::{HashMap, HashSet};

use super::*;

impl Program {
  /// Get the relations whose inputs, updates, and transitive dependencies are unchanged.
  pub fn persistent_relations(&self, new: &Self) -> HashSet<String> {
    self.persistent_relations_with_need_update_relations(new, &HashSet::new())
  }

  /// Get the reusable relations after also invalidating `need_update` and their dependents.
  pub fn persistent_relations_with_need_update_relations(
    &self,
    new: &Self,
    need_update: &HashSet<String>,
  ) -> HashSet<String> {
    if self.strata.is_empty() || new.strata.is_empty() {
      return HashSet::new();
    }

    let old_relations: HashMap<_, _> = self
      .relations()
      .map(|relation| (relation.predicate.as_str(), relation))
      .collect();
    let old_updates = updates_by_target(self);
    let new_updates = updates_by_target(new);
    let dependencies = new.dependency_graph();

    let changed = dependencies
      .node_indices()
      .filter(|&node| {
        let relation = dependencies[node];
        let predicate = relation.predicate.as_str();
        // Output and execution metadata do not participate in reuse.
        let same_inputs = old_relations.get(predicate).is_some_and(|old| {
          old.tuple_type == relation.tuple_type && old.input_file == relation.input_file && old.facts == relation.facts
        });
        need_update.contains(predicate) || !same_inputs || old_updates.get(predicate) != new_updates.get(predicate)
      })
      .collect();

    // The dependency graph points downstream, so this also invalidates every member of a changed cycle.
    let mut invalidated = Dfs::from_parts(changed, dependencies.visit_map());
    while invalidated.next(&dependencies).is_some() {}

    dependencies
      .node_indices()
      .filter(|node| !invalidated.discovered.is_visited(node))
      .map(|node| dependencies[node].predicate.clone())
      .collect()
  }
}

fn updates_by_target(program: &Program) -> HashMap<&str, Vec<&Dataflow>> {
  let mut by_target = HashMap::<_, Vec<_>>::new();
  for update in program.strata.iter().flat_map(|stratum| &stratum.updates) {
    by_target
      .entry(update.target.as_str())
      .or_default()
      .push(&update.dataflow);
  }
  // Compare updates independently of rule order without dropping duplicates.
  for updates in by_target.values_mut() {
    updates.sort_unstable();
  }
  by_target
}
