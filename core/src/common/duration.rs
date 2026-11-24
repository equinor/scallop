use jiff::{tz::Offset, SignedDuration, Span, Timestamp};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration {
  // Calendar months and elapsed time can have different signs.
  months: i32,
  elapsed: SignedDuration,
}

impl Duration {
  pub fn new(months: i32, elapsed: SignedDuration) -> Self {
    Self { months, elapsed }
  }

  pub fn years(years: i32) -> Self {
    Self::months(years.checked_mul(12).expect("Duration months overflow"))
  }

  pub fn months(months: i32) -> Self {
    Self::new(months, SignedDuration::ZERO)
  }

  pub fn days(days: i64) -> Self {
    Self::from(SignedDuration::from_secs(
      days.checked_mul(86_400).expect("Duration seconds overflow"),
    ))
  }
}

impl std::fmt::Display for Duration {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.months != 0 {
      write!(f, "R{}MO ", self.months)?;
    }
    if self.elapsed.is_zero() {
      return f.write_str("P0D");
    }
    if self.elapsed.is_negative() {
      f.write_str("-")?;
    }
    write!(f, "PT{}", self.elapsed.as_secs().unsigned_abs())?;
    let mut nanos = self.elapsed.subsec_nanos().unsigned_abs();
    if nanos != 0 {
      let mut width = 9;
      while nanos % 10 == 0 {
        nanos /= 10;
        width -= 1;
      }
      write!(f, ".{nanos:0width$}")?;
    }
    f.write_str("S")
  }
}

impl From<SignedDuration> for Duration {
  fn from(value: SignedDuration) -> Self {
    Self::new(0, value)
  }
}

impl std::ops::Add<Duration> for Duration {
  type Output = Duration;

  fn add(self, rhs: Duration) -> Self::Output {
    Self::new(
      self.months.checked_add(rhs.months).expect("Duration months overflow"),
      self.elapsed + rhs.elapsed,
    )
  }
}

impl std::ops::Sub<Duration> for Duration {
  type Output = Duration;

  fn sub(self, rhs: Duration) -> Self::Output {
    Self::new(
      self.months.checked_sub(rhs.months).expect("Duration months overflow"),
      self.elapsed - rhs.elapsed,
    )
  }
}

impl std::ops::Add<Duration> for Timestamp {
  type Output = Timestamp;

  fn add(self, rhs: Duration) -> Self::Output {
    shift_months(self, i64::from(rhs.months)) + rhs.elapsed
  }
}

impl std::ops::Sub<Duration> for Timestamp {
  type Output = Timestamp;

  fn sub(self, rhs: Duration) -> Self::Output {
    shift_months(self, -i64::from(rhs.months)) - rhs.elapsed
  }
}

fn shift_months(instant: Timestamp, months: i64) -> Timestamp {
  if months == 0 {
    return instant;
  }
  Span::new()
    .try_months(months)
    .and_then(|months| Offset::UTC.to_datetime(instant).checked_add(months))
    .and_then(|date| Offset::UTC.to_timestamp(date))
    .expect("DateTime month shift out of range")
}

impl std::ops::Mul<i32> for Duration {
  type Output = Duration;

  fn mul(self, rhs: i32) -> Self::Output {
    Self::new(
      self.months.checked_mul(rhs).expect("Duration months overflow"),
      self.elapsed * rhs,
    )
  }
}

impl std::ops::Div<i32> for Duration {
  type Output = Duration;

  fn div(self, rhs: i32) -> Self::Output {
    Self::new(self.months / rhs, self.elapsed / rhs)
  }
}

impl Serialize for Duration {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::*;
    let mut state = serializer.serialize_struct("Duration", 2)?;
    state.serialize_field("months", &self.months)?;
    state.serialize_field("duration", &self.elapsed.as_secs())?;
    state.end()
  }
}
