use indexmap::IndexSet;
use petgraph::unionfind::UnionFind;

use super::*;

pub fn deduplicate(ram: &mut Program) -> bool {
  let mut has_update = false;
  for stratum in &mut ram.strata {
    has_update |= deduplicate_within_stratum(stratum);
  }
  has_update
}

pub fn deduplicate_within_stratum(stratum: &mut Stratum) -> bool {
  let mut names = IndexSet::new();
  let candidates: Vec<_> = stratum
    .updates
    .iter()
    .filter(|update| update.target.starts_with("#temp#"))
    .map(|update| (names.insert_full(update.target.as_str()).0, &update.dataflow))
    .collect();
  let mut sets = UnionFind::new(names.len());
  for (i, &(left_id, left)) in candidates.iter().enumerate() {
    for &(right_id, right) in &candidates[i + 1..] {
      if left_id != right_id && left == right {
        sets.union(left_id, right_id);
      }
    }
  }

  // Keep the first target in update order, independently of union-find root selection.
  let mut representatives = vec![None; names.len()];
  let substitutions: Vec<_> = names
    .into_iter()
    .zip(sets.into_labeling())
    .filter_map(|(name, root)| {
      let representative = *representatives[root].get_or_insert(name);
      (name != representative).then(|| (name.to_owned(), representative.to_owned()))
    })
    .collect();
  for (alias, representative) in &substitutions {
    stratum.relations.remove(alias);
    stratum.updates.retain_mut(|update| {
      if update.target == *alias {
        false
      } else {
        update.dataflow.substitute_relation(alias, representative);
        true
      }
    });
  }
  !substitutions.is_empty()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::tuple_type::TupleType;

  #[test]
  fn deduplication_keeps_the_first_temp_in_each_group_and_rewrites_consumers() {
    let mut stratum = Stratum {
      is_recursive: false,
      relations: Default::default(),
      updates: vec![],
    };
    for (name, dataflow) in [
      ("#temp#3", Dataflow::relation("left")),
      ("#temp#5", Dataflow::relation("right")),
      ("#temp#1", Dataflow::relation("left")),
      ("#temp#2", Dataflow::relation("right")),
      ("#temp#4", Dataflow::relation("left")),
      ("visible", Dataflow::relation("left")),
      (
        "result",
        Dataflow::relation("#temp#1")
          .join(Dataflow::relation("#temp#2"))
          .join(Dataflow::relation("#temp#4")),
      ),
    ] {
      stratum.relations.insert(
        name.to_owned(),
        Relation::hidden_relation(name.to_owned(), TupleType::unit()),
      );
      stratum.updates.push(Update {
        target: name.to_owned(),
        dataflow,
      });
    }

    assert!(deduplicate_within_stratum(&mut stratum));
    assert_eq!(
      stratum.relations.keys().map(String::as_str).collect::<Vec<_>>(),
      ["#temp#3", "#temp#5", "result", "visible"]
    );
    assert_eq!(
      stratum
        .updates
        .iter()
        .map(|update| update.target.as_str())
        .collect::<Vec<_>>(),
      ["#temp#3", "#temp#5", "visible", "result"]
    );
    assert_eq!(
      stratum.updates.last().unwrap().dataflow,
      Dataflow::relation("#temp#3")
        .join(Dataflow::relation("#temp#5"))
        .join(Dataflow::relation("#temp#3"))
    );
    assert!(!deduplicate_within_stratum(&mut stratum));
  }
}
