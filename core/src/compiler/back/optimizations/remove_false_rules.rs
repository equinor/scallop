use super::super::*;

pub fn remove_false_rules(rules: &mut Vec<Rule>) {
  rules.retain(|rule| rule.body_literals().find(|l| matches!(l, Literal::False)).is_none())
}
