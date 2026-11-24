#[derive(Default)]
pub enum PredicateSet {
  None,
  #[default]
  All,
  Some(Vec<String>),
}

impl PredicateSet {
  pub fn contains(&self, p: &String) -> bool {
    match self {
      Self::None => false,
      Self::All => true,
      Self::Some(ps) => ps.contains(p),
    }
  }
}
