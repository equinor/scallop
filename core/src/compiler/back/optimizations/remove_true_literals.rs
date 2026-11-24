use super::super::*;

pub fn remove_true_literals(rule: &mut Rule) {
  rule.body.args.retain(|l| !matches!(l, Literal::True))
}
