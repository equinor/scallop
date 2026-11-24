use std::collections::*;

use crate::compiler::front::*;
use crate::runtime::database::StorageMetadata;

#[derive(Clone, Debug)]
pub struct StorageAttributeAnalysis {
  pub storage_attrs: HashMap<String, StorageMetadata>,
  pub errors: Vec<FrontCompileErrorMessage>,
}

impl StorageAttributeAnalysis {
  pub fn new() -> Self {
    Self {
      storage_attrs: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn get_storage(&self, rel: &str) -> Option<&StorageMetadata> {
    self.storage_attrs.get(rel)
  }

  pub fn find_storage_attr(&mut self, attrs: &[Attribute]) -> Option<StorageMetadata> {
    if let Some(attr) = attrs.iter().find(|attr| attr.attr_name() == "storage") {
      if let Some(storage_arg) = attr.pos_arg(0) {
        match storage_arg.as_string() {
          Some(storage_name) => {
            use std::str::FromStr;
            match StorageMetadata::from_str(&storage_name) {
              Ok(metadata) => Some(metadata),
              Err(_) => {
                self.errors.push(
                  FrontCompileErrorMessage::error()
                    .msg(format!(
                      "Cannot parse storage type `{}` for @storage. Choose from {}",
                      storage_name,
                      StorageMetadata::choices_string()
                    ))
                    .src(storage_arg.location().clone()),
                );
                None
              }
            }
          }
          None => {
            self.errors.push(
              FrontCompileErrorMessage::error()
                .msg("Expected a string for the first positional argument for @storage".to_string())
                .src(attr.location().clone()),
            );
            None
          }
        }
      } else {
        self.errors.push(
          FrontCompileErrorMessage::error()
            .msg("Needs at least one positional argument for @storage".to_string())
            .src(attr.location().clone()),
        );
        None
      }
    } else {
      None
    }
  }
}

impl Default for StorageAttributeAnalysis {
  fn default() -> Self {
    Self::new()
  }
}

impl NodeVisitor<RelationDecl> for StorageAttributeAnalysis {
  fn visit(&mut self, node: &RelationDecl) {
    if let Some(storage) = self.find_storage_attr(node.attrs()) {
      for pred in node.head_predicates() {
        if !self.storage_attrs.contains_key(&pred) {
          self.storage_attrs.insert(pred.clone(), storage.clone());
        } else {
          self.errors.push(
            FrontCompileErrorMessage::error()
              .msg(format!("@storage double specified on relation `{}`", pred))
              .src(node.location().clone()),
          );
          return;
        }
      }
    }
  }
}

impl NodeVisitor<RelationTypeDecl> for StorageAttributeAnalysis {
  fn visit(&mut self, node: &RelationTypeDecl) {
    if let Some(storage) = self.find_storage_attr(node.attrs()) {
      for rel_type in node.rel_types() {
        let pred = rel_type.predicate_name();
        if !self.storage_attrs.contains_key(pred) {
          self.storage_attrs.insert(pred.clone(), storage.clone());
        } else {
          self.errors.push(
            FrontCompileErrorMessage::error()
              .msg(format!("@storage double specified on relation `{}`", pred))
              .src(node.location().clone()),
          );
          return;
        }
      }
    }
  }
}
