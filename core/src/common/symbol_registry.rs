use indexmap::IndexSet;

#[derive(Clone, Debug, Default)]
pub struct SymbolRegistry {
  symbols: IndexSet<String>,
}

impl SymbolRegistry {
  /// Create a new symbol registry
  pub fn new() -> Self {
    Self::default()
  }

  /// Check if the registry is empty
  pub fn is_empty(&self) -> bool {
    self.symbols.is_empty()
  }

  /// Register a symbol into the registry and return an ID to represent the symbol
  pub fn register(&mut self, symbol: String) -> usize {
    self.symbols.insert_full(symbol).0
  }

  /// Getting the symbol id using the string
  pub fn get_id(&self, symbol: &str) -> Option<usize> {
    self.symbols.get_index_of(symbol)
  }

  /// Getting a symbol using its ID
  pub fn get_symbol(&self, id: usize) -> Option<&String> {
    self.symbols.get_index(id)
  }
}
