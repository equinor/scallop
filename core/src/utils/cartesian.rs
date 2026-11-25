use itertools::Itertools;

pub fn cartesian(sizes: Vec<usize>) -> Vec<Vec<usize>> {
  cartesian_indices(sizes.into_iter()).collect()
}

pub(crate) fn cartesian_indices(sizes: impl DoubleEndedIterator<Item = usize>) -> impl Iterator<Item = Vec<usize>> {
  // Scallop advances the first dimension fastest; itertools advances the last.
  sizes
    .rev()
    .map(|size| 0..size)
    .multi_cartesian_product()
    .map(|mut row| {
      row.reverse();
      row
    })
}

#[cfg(test)]
mod tests {
  use super::cartesian;

  #[test]
  fn first_dimension_advances_fastest() {
    assert_eq!(
      cartesian(vec![2, 3]),
      vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1], vec![0, 2], vec![1, 2]]
    );
    assert_eq!(cartesian(vec![]), vec![Vec::<usize>::new()]);
    assert!(cartesian(vec![2, 0, 3]).is_empty());
  }
}
