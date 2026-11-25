#[derive(Clone)]
pub(crate) struct JoinProduct<L, R> {
  left: Vec<L>,
  right: Vec<R>,
  i: usize,
  j: usize,
}

impl<L, R> JoinProduct<L, R> {
  pub(crate) fn new(left: Vec<L>, right: Vec<R>) -> Self {
    Self {
      left,
      right,
      i: 0,
      j: 0,
    }
  }

  pub(crate) fn group_lengths(&self) -> (usize, usize) {
    (self.left.len(), self.right.len())
  }
}

impl<L: Clone, R: Clone> Iterator for JoinProduct<L, R> {
  type Item = (L, R);

  fn next(&mut self) -> Option<Self::Item> {
    let left = self.left.get(self.i)?;
    let right = self.right.get(self.j)?;
    self.j += 1;
    if self.j == self.right.len() {
      self.i += 1;
      self.j = 0;
    }
    Some((left.clone(), right.clone()))
  }
}

#[cfg(test)]
mod tests {
  use super::JoinProduct;

  #[test]
  fn iterates_left_major_and_clones_the_current_position() {
    let mut product = JoinProduct::new(vec![1, 2], vec!['a', 'b', 'c']);
    assert_eq!(product.next(), Some((1, 'a')));
    let remaining = [(1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (2, 'c')];
    assert!(product.clone().eq(remaining));
    assert!(product.eq(remaining));
  }
}
