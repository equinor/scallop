use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A float key for collections: signed zeros are equal, and all NaNs sort last and compare equal.
#[derive(Clone, Copy)]
pub(crate) struct FloatKey<T>(pub T);

macro_rules! impl_float_key {
  ($float:ty, $bits:ty) => {
    impl FloatKey<$float> {
      fn canonical(self) -> $float {
        if self.0 == 0.0 {
          0.0
        } else if self.0.is_nan() {
          <$float>::NAN.abs()
        } else {
          self.0
        }
      }
    }

    impl PartialEq for FloatKey<$float> {
      fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
      }
    }

    impl Eq for FloatKey<$float> {}

    impl PartialOrd for FloatKey<$float> {
      fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
      }
    }

    impl Ord for FloatKey<$float> {
      fn cmp(&self, other: &Self) -> Ordering {
        self.canonical().total_cmp(&other.canonical())
      }
    }

    impl Hash for FloatKey<$float> {
      fn hash<H: Hasher>(&self, state: &mut H) {
        <$bits>::from_ne_bytes(self.canonical().to_ne_bytes()).hash(state);
      }
    }
  };
}

impl_float_key!(f32, i32);
impl_float_key!(f64, i64);

/// Floating Point trait (f32, f64)
pub trait Float:
  Sized
  + Copy
  + Clone
  + PartialEq
  + PartialOrd
  + std::fmt::Debug
  + std::fmt::Display
  + std::ops::Add<Self, Output = Self>
  + std::ops::Sub<Self, Output = Self>
  + std::ops::Mul<Self, Output = Self>
  + std::ops::Div<Self, Output = Self>
  + TryInto<f64>
{
  fn zero() -> Self;

  fn one() -> Self;

  fn from_f64(f: f64) -> Self;
}

impl Float for f32 {
  fn zero() -> Self {
    0.0
  }

  fn one() -> Self {
    1.0
  }

  fn from_f64(f: f64) -> Self {
    f as f32
  }
}

impl Float for f64 {
  fn zero() -> Self {
    0.0
  }

  fn one() -> Self {
    1.0
  }

  fn from_f64(f: f64) -> Self {
    f
  }
}
