use itertools::{EitherOrBoth, Itertools};

use super::super::*;

/// The DualNumber that contains the real and gradient, where the gradient
/// is sparse
///
/// Note: The difference between dual_number and dual_number_2 is that
/// dual_number_2 supports dynamic resize. That is, when performing binary
/// operations like + or *, we do not require the gradient to be coming
/// from the same dimensionality. This is useful for many provenance
/// structures as in the beginning, the information of the number of input
/// variables is unknown.
#[derive(Clone)]
pub struct DualNumber2 {
  pub real: f64,
  pub gradient: Gradient,
}

impl Tag for DualNumber2 {}

impl DualNumber2 {
  pub fn new(id: usize, real: f64) -> Self {
    Self {
      real,
      gradient: Gradient::singleton(id),
    }
  }

  pub fn one() -> Self {
    Self {
      real: 1.0,
      gradient: Gradient::empty(),
    }
  }

  pub fn zero() -> Self {
    Self {
      real: 0.0,
      gradient: Gradient::empty(),
    }
  }

  pub fn constant(real: f64) -> Self {
    Self {
      real,
      gradient: Gradient::empty(),
    }
  }

  pub fn clamp_real(&mut self) {
    self.real = self.real.clamp(0.0, 1.0);
  }

  pub fn max(&self, other: &Self) -> Self {
    if self.real > other.real {
      self.clone()
    } else {
      other.clone()
    }
  }

  pub fn min(&self, other: &Self) -> Self {
    if self.real < other.real {
      self.clone()
    } else {
      other.clone()
    }
  }
}

impl std::ops::Neg for &DualNumber2 {
  type Output = DualNumber2;

  fn neg(self) -> Self::Output {
    Self::Output {
      real: 1.0 - self.real,
      gradient: -&self.gradient,
    }
  }
}

impl std::fmt::Debug for DualNumber2 {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("DualNumber2")
      .field("real", &self.real)
      .field("gradient", &self.gradient)
      .finish()
  }
}

impl std::fmt::Display for DualNumber2 {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.real))
  }
}

impl<'a> std::ops::Mul<&'a DualNumber2> for &'a DualNumber2 {
  type Output = DualNumber2;

  fn mul(self, rhs: &'a DualNumber2) -> Self::Output {
    Self::Output {
      real: self.real * rhs.real,
      gradient: &self.gradient * &rhs.real + &(&rhs.gradient * &self.real),
    }
  }
}

impl<'a> std::ops::Add<&'a DualNumber2> for &'a DualNumber2 {
  type Output = DualNumber2;

  fn add(self, rhs: &'a DualNumber2) -> Self::Output {
    Self::Output {
      real: self.real + rhs.real,
      gradient: self.gradient.clone() + &rhs.gradient,
    }
  }
}

#[derive(Clone, Debug)]
pub struct Gradient {
  pub indices: Vec<usize>,
  pub values: Vec<f64>,
}

impl Gradient {
  pub fn empty() -> Self {
    Self {
      indices: vec![],
      values: vec![],
    }
  }

  pub fn singleton(id: usize) -> Self {
    Self {
      indices: vec![id],
      values: vec![1.0],
    }
  }
}

impl<'a> std::ops::Mul<&'a f64> for &'a Gradient {
  type Output = Gradient;

  fn mul(self, rhs: &'a f64) -> Self::Output {
    Self::Output {
      indices: self.indices.clone(),
      values: self.values.iter().map(|v| v * rhs).collect(),
    }
  }
}

impl<'a> std::ops::Add<&'a Gradient> for Gradient {
  type Output = Self;

  fn add(self, rhs: &'a Gradient) -> Self::Output {
    let new_capacity = self.indices.len().max(rhs.indices.len());
    let mut new_indices = Vec::with_capacity(new_capacity);
    let mut new_values = Vec::with_capacity(new_capacity);

    let left = self
      .indices
      .into_iter()
      .enumerate()
      .map(|(i, index)| (index, self.values[i]));
    let right = rhs.indices.iter().enumerate().map(|(i, &index)| (index, rhs.values[i]));
    for pair in left.merge_join_by(right, |(left, _), (right, _)| left.cmp(right)) {
      let (index, value) = match pair {
        EitherOrBoth::Left(entry) | EitherOrBoth::Right(entry) => entry,
        EitherOrBoth::Both((index, left), (_, right)) => (index, left + right),
      };
      new_indices.push(index);
      new_values.push(value);
    }

    Self::Output {
      indices: new_indices,
      values: new_values,
    }
  }
}

impl std::ops::Neg for &Gradient {
  type Output = Gradient;

  fn neg(self) -> Self::Output {
    Self::Output {
      indices: self.indices.clone(),
      values: self.values.iter().map(|v| -v).collect(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::DualNumber2;

  #[test]
  fn product_merges_overlapping_and_disjoint_sparse_gradient_entries() {
    let first = &DualNumber2::new(1, 0.125) + &DualNumber2::new(42, 0.25);
    let second = &DualNumber2::new(42, 0.375) + &DualNumber2::new(usize::MAX, 0.125);
    let result = &first * &second;
    assert_eq!(result.real, 0.1875);
    assert_eq!(result.gradient.indices, [1, 42, usize::MAX]);
    assert_eq!(result.gradient.values, [0.5, 0.875, 0.375]);
  }
}
