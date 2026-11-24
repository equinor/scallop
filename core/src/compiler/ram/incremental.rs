use petgraph::graph::NodeIndex;
use petgraph::visit::*;
use petgraph::EdgeDirection;
use std::collections::*;

use crate::common::input_file::InputFile;
use crate::common::tuple_type::TupleType;

use super::*;

impl Program {
  /// Get the set of relations (names) that can be reused from an old program.
  ///
  /// This is done through comparing the SCC in topological order. For each pair
  /// of SCCs in both old and new program, if they are the same, then we say
  /// every relation in this SCC can be persisted (reused).
  pub fn persistent_relations(&self, new: &Self) -> HashSet<String> {
    self.persistent_relations_with_need_update_relations(new, &HashSet::new())
  }

  /// Get the set of relations (names) that can be reused from an old program, given an additional set of relations marked as need update (cannot be persisted)
  ///
  /// This is done through comparing the SCC in topological order. For each pair
  /// of SCCs in both old and new program, if they are the same, then we say
  /// every relation in this SCC can be persisted (reused).
  pub fn persistent_relations_with_need_update_relations(
    &self,
    new: &Self,
    need_update: &HashSet<String>,
  ) -> HashSet<String> {
    let mut can_increment = HashSet::new();

    // Generate context
    let (cg1, mut fringe_1, mut visited_1) = scc(self);
    let (cg2, mut fringe_2, mut visited_2) = scc(new);

    // Iterate through the fringes
    while !fringe_1.is_empty() && !fringe_2.is_empty() {
      let (s1, n1) = fringe_1.pop_first().unwrap();
      let (s2, n2) = fringe_2.pop_first().unwrap();

      // Compare the two strata
      match s1.cmp(&s2) {
        std::cmp::Ordering::Equal => {
          // First check if s1 and s2 are derived with the same updates
          let u1 = incoming_updates(n1, &cg1);
          let u2 = incoming_updates(n2, &cg2);
          if u1 != u2 {
            continue;
          }

          // Check if the stratum relations contain anything in the `need_update` set
          if !s1.iter().any(|relation| need_update.contains(relation.predicate)) {
            // First add the relations to `can_increment`
            for relation in s1 {
              can_increment.insert(relation.predicate.to_owned());
            }

            // Add to visited
            visited_1.insert(n1);
            visited_2.insert(n2);
          }

          // Then add the next nodes of n1
          for out_edge in cg1.edges(n1) {
            let target = out_edge.target();
            if incoming_node_all_visited(target, &cg1, &visited_1) {
              fringe_1.insert((persistence_key(&cg1[target]), target));
            }
          }

          // Add the next nodes of n2
          for out_edge in cg2.edges(n2) {
            let target = out_edge.target();
            if incoming_node_all_visited(target, &cg2, &visited_2) {
              fringe_2.insert((persistence_key(&cg2[target]), target));
            }
          }
        }
        std::cmp::Ordering::Greater => {
          fringe_2.insert((s2, n2));
        }
        std::cmp::Ordering::Less => {
          fringe_1.insert((s1, n1));
        }
      }
    }

    can_increment
  }
}

// Reuse compares relation inputs, not the output and execution metadata in Relation::cmp.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct RelationPersistenceKey<'a> {
  predicate: &'a str,
  tuple_type: &'a TupleType,
  input_file: &'a Option<InputFile>,
  facts: &'a [Fact],
}

fn persistence_key<'a>(relations: &[&'a Relation]) -> Vec<RelationPersistenceKey<'a>> {
  relations
    .iter()
    .map(|r| RelationPersistenceKey {
      predicate: &r.predicate,
      tuple_type: &r.tuple_type,
      input_file: &r.input_file,
      facts: &r.facts,
    })
    .collect()
}

type Fringe<'a> = BTreeSet<(Vec<RelationPersistenceKey<'a>>, NodeIndex)>;

type Visited = HashSet<NodeIndex>;

fn scc<'a>(p: &'a Program) -> (RamDependencySCCGraph<'a>, Fringe<'a>, Visited) {
  let scc = p.scc();
  let fringe = fringe(&scc);
  let visited = visited_nodes(&fringe);
  (scc, fringe, visited)
}

fn fringe<'a>(g: &RamDependencySCCGraph<'a>) -> Fringe<'a> {
  g.node_indices()
    .filter_map(|n| {
      let mut in_edges = g.edges_directed(n, petgraph::EdgeDirection::Incoming);
      if in_edges.next().is_none() {
        Some((persistence_key(&g[n]), n))
      } else {
        None
      }
    })
    .collect()
}

fn visited_nodes(f: &Fringe<'_>) -> HashSet<NodeIndex> {
  f.iter().map(|(_, n)| *n).collect()
}

fn incoming_updates<'a>(n: NodeIndex, g: &RamDependencySCCGraph<'a>) -> BTreeSet<&'a Update> {
  g.edges_directed(n, EdgeDirection::Incoming)
    .flat_map(|e| e.weight().clone().into_iter())
    .collect()
}

fn incoming_node_all_visited(n: NodeIndex, g: &RamDependencySCCGraph, visited: &HashSet<NodeIndex>) -> bool {
  g.edges_directed(n, EdgeDirection::Incoming)
    .all(|e| visited.contains(&e.source()))
}
