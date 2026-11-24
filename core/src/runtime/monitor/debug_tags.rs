use crate::common::tuple::Tuple;
use crate::runtime::provenance::Provenance;

use super::*;

#[derive(Clone)]
pub struct DebugTagsMonitor;

impl<Prov: Provenance> Monitor<Prov> for DebugTagsMonitor {
  fn name(&self) -> &'static str {
    "debug-tags"
  }

  fn observe_loading_relation(&self, relation: &str) {
    println!("[Tagging Relation] {}", relation)
  }

  fn observe_tagging(&self, _tup: &Tuple, input_tag: &Option<Prov::InputTag>, _tag: &Prov::Tag) {
    if input_tag.is_some() {
      println!("[Tagging] Applied an explicit input tag")
    } else {
      println!("[Tagging] Applied a default tag")
    }
  }

  fn observe_recovering_relation(&self, relation: &str) {
    println!("[Recover Relation] {}", relation)
  }

  fn observe_recover(&self, _tup: &Tuple, _tag: &Prov::Tag, _output_tag: &Prov::OutputTag) {
    println!("[Recover] Recovered an output tag")
  }
}
