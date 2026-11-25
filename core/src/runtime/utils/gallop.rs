pub(crate) fn gallop_index<T>(slice: &[T], mut begin: usize, mut cmp: impl FnMut(&T) -> bool) -> usize {
  begin = begin.min(slice.len());
  if slice.get(begin).is_some_and(&mut cmp) {
    let mut step = 1;
    while step < slice.len() - begin && cmp(&slice[begin + step]) {
      begin += step;
      step <<= 1;
    }

    step >>= 1;
    while step > 0 {
      if step < slice.len() - begin && cmp(&slice[begin + step]) {
        begin += step;
      }
      step >>= 1;
    }

    begin += 1; // advance one, as we always stayed < value
  }

  begin
}

#[cfg(test)]
mod tests {
  use super::gallop_index;

  #[test]
  fn gallop_respects_the_start_cursor_and_skips_long_runs() {
    let rows = (0..4096).collect::<Vec<_>>();
    assert_eq!(gallop_index(&rows, 50, |row| *row < 40), 50);
    assert_eq!(gallop_index(&rows, rows.len(), |row| *row < 5000), rows.len());
    let mut comparisons = 0;
    let index = gallop_index(&rows, 50, |row| {
      comparisons += 1;
      *row < 4000
    });
    assert_eq!(index, 4000);
    assert!(comparisons <= 26, "{comparisons} comparisons");
  }
}
