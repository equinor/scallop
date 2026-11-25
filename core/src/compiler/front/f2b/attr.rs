use super::super::*;
use crate::compiler::back;

impl FrontContext {
  pub fn back_relation_attributes(&self, relation: &String) -> back::Attributes {
    let mut attrs = back::Attributes::new();

    // Check demand attributes
    if let Some(pattern) = self.analysis.demand_attr_analysis.demand_pattern(relation) {
      attrs.insert(back::attributes::DemandAttribute::new(pattern.clone()));
    }

    // Check input files
    if let Some(input_file) = self.analysis.input_files_analysis.input_file(relation) {
      attrs.insert(back::attributes::InputFileAttribute::new(input_file.clone()));
    }

    // Check goal predicates
    if self.analysis.goal_relation_analysis.is_goal(relation) {
      attrs.insert(back::attributes::GoalAttribute);
    }

    // Check scheduler predicates
    if let Some(scheduler) = self.analysis.scheduler_attr_analysis.get_scheduler(relation) {
      attrs.insert(back::attributes::SchedulerAttribute::new(scheduler.clone()));
    }

    // Check storage predicates
    if let Some(storage) = self.analysis.storage_attr_analysis.get_storage(relation) {
      attrs.insert(back::attributes::StorageAttribute::new(storage.clone()));
    }

    attrs
  }
}
