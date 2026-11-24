use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// An ID allocator
#[derive(Debug, Clone, Default)]
pub struct IdAllocator {
  id: usize,
}

impl IdAllocator {
  pub fn new() -> Self {
    Self { id: 0 }
  }

  pub fn alloc(&mut self) -> usize {
    let result = self.id;
    self.id += 1;
    result
  }
}

/// An alternative ID allocator which has internal mutability
#[derive(Debug, Clone, Default)]
pub struct IdAllocator2 {
  next: Arc<AtomicUsize>,
}

impl IdAllocator2 {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn new_with_start(start: usize) -> Self {
    Self {
      next: Arc::new(AtomicUsize::new(start)),
    }
  }

  pub fn alloc(&self) -> usize {
    self
      .next
      .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
      .expect("ID allocator exhausted")
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn shared_allocations_do_not_repeat_or_wrap() {
    let allocator = IdAllocator2::new_with_start(usize::MAX - 2);
    let clone = allocator.clone();
    let other = std::thread::spawn(move || clone.alloc());
    let mut ids = [allocator.alloc(), other.join().unwrap()];
    ids.sort_unstable();
    assert_eq!(ids, [usize::MAX - 2, usize::MAX - 1]);
    assert!(std::panic::catch_unwind(|| allocator.alloc()).is_err());
  }
}
