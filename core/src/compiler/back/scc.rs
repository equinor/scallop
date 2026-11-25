use colored::*;
use petgraph::{
  graph::{EdgeIndex, Graph, NodeIndex},
  visit::EdgeRef,
};
use std::collections::*;
use std::sync::OnceLock;

use crate::compiler::back::attributes::GoalAttribute;

use super::ast::*;

/// The type of a dependency graph edge: positive, negative, or aggregation
#[derive(Debug, Clone)]
pub enum DependencyGraphEdge {
  Positive,
  Negative,
  Aggregation,
}

impl std::fmt::Display for DependencyGraphEdge {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Positive => f.write_str("positive"),
      Self::Negative => f.write_str("negative"),
      Self::Aggregation => f.write_str("aggregation"),
    }
  }
}

impl DependencyGraphEdge {
  /// Check if the edge needs to be stratified
  pub fn needs_stratification(&self) -> bool {
    !matches!(self, Self::Positive)
  }
}

/// A graph storing the dependencies between predicates
#[derive(Clone)]
pub struct DependencyGraph {
  graph: Graph<String, DependencyGraphEdge>,
  predicate_to_node_id: HashMap<String, NodeIndex>,
  sccs: OnceLock<Vec<Vec<NodeIndex>>>,
}

impl DependencyGraph {
  /// Create an empty dependency graph
  pub fn new() -> Self {
    Self {
      graph: Graph::new(),
      predicate_to_node_id: HashMap::new(),
      sccs: OnceLock::new(),
    }
  }

  /// Add a predicate to the dependency graph
  pub fn add_predicate(&mut self, predicate: &String) -> NodeIndex {
    if let Some(ni) = self.predicate_to_node_id.get(predicate) {
      *ni
    } else {
      self.sccs.take();
      let node_id = self.graph.add_node(predicate.clone());
      self.predicate_to_node_id.insert(predicate.clone(), node_id);
      node_id
    }
  }

  /// Get the node id of a predicate
  pub fn predicate_node(&self, predicate: &String) -> NodeIndex {
    *self.predicate_to_node_id.get(predicate).unwrap()
  }

  /// Add a dependency between two predicates
  pub fn add_dependency(&mut self, src: &String, dst: &String, edge: DependencyGraphEdge) -> EdgeIndex {
    let src_id = self.predicate_to_node_id[src];
    let dst_id = self.predicate_to_node_id[dst];
    self.sccs.take();
    self.graph.add_edge(src_id, dst_id, edge)
  }

  pub fn unused_relations(&self, targets: &HashSet<String>) -> HashSet<String> {
    let mut unrelated = HashSet::new();
    let output_ids = targets.iter().map(|r| self.predicate_node(r)).collect::<Vec<_>>();
    for (predicate, rid) in &self.predicate_to_node_id {
      if !targets.contains(predicate) {
        let connected_to_output = output_ids
          .iter()
          .any(|oid| petgraph::algo::has_path_connecting(&self.graph, *oid, *rid, None));
        if !connected_to_output {
          unrelated.insert(predicate.clone());
        }
      }
    }
    unrelated
  }

  pub fn compute_scc(&mut self) {
    self.sccs();
  }

  fn sccs(&self) -> &[Vec<NodeIndex>] {
    self.sccs.get_or_init(|| petgraph::algo::kosaraju_scc(&self.graph))
  }

  pub fn stratify(&self) -> Result<Vec<Stratum>, SCCError> {
    let sccs = self.sccs();
    for scc in sccs {
      for node_id in scc {
        for edge in self.graph.edges(*node_id) {
          if edge.weight().needs_stratification() && scc.contains(&edge.target()) {
            return Err(SCCError::CannotStratify {
              pred_1: self.graph[edge.source()].clone(),
              pred_2: self.graph[edge.target()].clone(),
              edge: edge.weight().clone(),
            });
          }
        }
      }
    }

    // Kosaraju returns SCCs in reverse topological order, so dependencies come first.
    Ok(
      sccs
        .iter()
        .map(|scc| Stratum {
          predicates: scc.iter().map(|&node| self.graph[node].clone()).collect(),
          is_recursive: scc.len() > 1 || self.graph.contains_edge(scc[0], scc[0]),
        })
        .collect(),
    )
  }
}

impl Default for DependencyGraph {
  fn default() -> Self {
    Self::new()
  }
}

#[derive(Debug, Clone)]
pub struct Stratum {
  pub predicates: Vec<String>,
  pub is_recursive: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SCCError {
  #[error("{} Cannot stratify program: {edge} cycle detected between predicate `{pred_1}` and `{pred_2}`", "[Error]".red())]
  CannotStratify {
    pred_1: String,
    pred_2: String,
    edge: DependencyGraphEdge,
  },
}

impl std::fmt::Debug for DependencyGraph {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let sccs = self.sccs.get().map_or(&[][..], Vec::as_slice);
    for (i, scc) in sccs.iter().enumerate() {
      f.write_fmt(format_args!("SCC#{}: {{", i))?;
      f.write_str(
        &scc
          .iter()
          .map(|j| self.graph[*j].clone())
          .collect::<Vec<_>>()
          .join(", "),
      )?;
      f.write_str("}")?;
      if i < sccs.len() - 1 {
        f.write_str(", ")?;
      }
    }
    Ok(())
  }
}

impl Program {
  pub fn dependency_graph(&self) -> DependencyGraph {
    type E = DependencyGraphEdge;

    // The graph to work on
    let mut graph = DependencyGraph::new();

    // First add all the nodes
    for relation in &self.relations {
      graph.add_predicate(&relation.predicate);
    }

    // Then add all the dependencies by going through rules
    for rule in &self.rules {
      // Collect the head predicate
      let head_predicate = rule.head_predicate();

      // Step 1. Deal with the dependencies between goal predicate and its dependencies
      if let Some(head_relation) = self.relation_of_predicate(head_predicate)
        && head_relation.attributes.get::<GoalAttribute>().is_some()
      {
        for atom in rule.body_literals() {
          match atom {
            Literal::Atom(a) if !self.predicate_registry.contains(&a.predicate) => {
              graph.add_dependency(&a.predicate, head_predicate, E::Positive);
            }
            _ => {}
          }
        }
      }

      // Step 2. Deal with dependencies related to Entity and Functors
      // Collect all the related functor predicates
      let does_create_dyn_ent = rule.needs_dynamically_parse_entity(&self.function_registry, &self.predicate_registry);
      let functor_predicates: Vec<_> = if does_create_dyn_ent {
        // If needs dynamically parse entity, then all entities could be mentioned
        // therefore we pull all the adt variants from the registry
        self
          .adt_variant_registry
          .iter()
          .map(|(_, v)| &v.relation_name)
          .collect()
      } else {
        // Otherwise, we just collect all the `new` expression functors occurred
        // in the rule itself
        rule.collect_new_expr_functors().collect()
      };

      // Functor predicates depends on head
      for functor_predicate in &functor_predicates {
        graph.add_dependency(functor_predicate, head_predicate, E::Positive);
        graph.add_dependency(head_predicate, functor_predicate, E::Positive);
      }

      // A recording dependency helper function
      let mut record_dependency = |pred, edge_type: E| {
        graph.add_dependency(head_predicate, pred, edge_type.clone());
        for functor_predicate in &functor_predicates {
          graph.add_dependency(functor_predicate, pred, edge_type.clone());
        }
      };
      for atom in rule.body_literals() {
        match atom {
          Literal::Atom(a) if !self.predicate_registry.contains(&a.predicate) => {
            record_dependency(&a.predicate, E::Positive);
          }
          Literal::NegAtom(a) if !self.predicate_registry.contains(&a.atom.predicate) => {
            record_dependency(&a.atom.predicate, E::Negative);
          }
          Literal::Reduce(r) => {
            let reduce_predicate = &r.body_formula.predicate;
            record_dependency(reduce_predicate, E::Aggregation);

            // Add group by predicate also as an aggregation dependency
            if let Some(group_by_atom) = &r.group_by_formula {
              let group_by_predicate = &group_by_atom.predicate;
              record_dependency(group_by_predicate, E::Aggregation);
            }
          }
          _ => {}
        }
      }
    }

    // Return the graph
    graph
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn stratification_recomputes_modified_components() {
    let mut graph = DependencyGraph::new();
    let consumer = "consumer".to_owned();
    let dependency = "dependency".to_owned();
    graph.add_predicate(&consumer);
    graph.add_predicate(&dependency);
    graph.add_dependency(&consumer, &dependency, DependencyGraphEdge::Positive);
    graph.add_dependency(&consumer, &consumer, DependencyGraphEdge::Positive);

    graph.compute_scc();
    let strata = graph.stratify().unwrap();
    assert_eq!(strata[0].predicates, ["dependency"]);
    assert!(!strata[0].is_recursive);
    assert_eq!(strata[1].predicates, ["consumer"]);
    assert!(strata[1].is_recursive);

    graph.add_dependency(&dependency, &consumer, DependencyGraphEdge::Negative);
    assert!(matches!(
      graph.stratify(),
      Err(SCCError::CannotStratify { pred_1, pred_2, edge: DependencyGraphEdge::Negative })
        if pred_1 == dependency && pred_2 == consumer
    ));
  }
}
