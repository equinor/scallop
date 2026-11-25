use itertools::Itertools;

use super::tuple::*;

pub trait Tuples<'a> {
  fn sort(self, num_args: usize) -> Vec<&'a Tuple>;

  fn minimum(self) -> Vec<&'a Tuple>;

  fn arg_minimum(self, num_args: usize) -> Vec<&'a Tuple>;

  fn maximum(self) -> Vec<&'a Tuple>;

  fn arg_maximum(self, num_args: usize) -> Vec<&'a Tuple>;
}

impl<'a, I> Tuples<'a> for I
where
  I: Iterator<Item = &'a Tuple>,
{
  fn sort(self, num_args: usize) -> Vec<&'a Tuple> {
    let mut collected = self.collect::<Vec<_>>();
    collected.sort_by_key(|e| &e[num_args..]);
    collected
  }

  fn minimum(self) -> Vec<&'a Tuple> {
    self.min_set()
  }

  fn arg_minimum(self, num_args: usize) -> Vec<&'a Tuple> {
    self.min_set_by_key(|v| &v[num_args..])
  }

  fn maximum(self) -> Vec<&'a Tuple> {
    self.max_set()
  }

  fn arg_maximum(self, num_args: usize) -> Vec<&'a Tuple> {
    self.max_set_by_key(|v| &v[num_args..])
  }
}
