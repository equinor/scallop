use std::fmt::Display;
use std::sync::LazyLock;

use jiff::civil::{DateTime as CivilDateTime, Time};
use jiff::fmt::{rfc2822, strtime::BrokenDownTime};
use jiff::tz::Offset;
use jiff::{SignedDuration, Span, SpanRelativeTo, Timestamp};
use num_bigint::BigInt;
use regex::Regex;

use crate::common::duration::Duration;

/// Parse a UTC instant. A missing time defaults to midnight, and a missing date defaults to today.
pub fn parse_date_time_string(input: &str) -> Option<Timestamp> {
  let input = input.trim();
  if let Ok(timestamp) = input.parse() {
    return Some(timestamp);
  }
  if let Ok(timestamp) = rfc2822::DateTimeParser::new().parse_timestamp(input) {
    return Some(timestamp);
  }
  if input.bytes().all(|b| b.is_ascii_digit()) {
    match input.len() {
      10 => return Timestamp::from_second(input.parse().ok()?).ok(),
      13 => return Timestamp::from_millisecond(input.parse().ok()?).ok(),
      19 => return Timestamp::from_nanosecond(i128::from(input.parse::<i64>().ok()?)).ok(),
      _ => {}
    }
  }

  if let Ok((date, end)) = BrokenDownTime::parse_prefix("%Y-%m-%d", input) {
    if let Some(offset) = parse_offset(input[end..].trim_start()) {
      return offset.to_timestamp(date.to_datetime().ok()?).ok();
    }
  }

  let (input, offset) = input
    .rsplit_once(char::is_whitespace)
    .and_then(|(date, zone)| Some((date.trim_end(), parse_offset(zone)?)))
    .unwrap_or((input, Offset::UTC));
  offset.to_timestamp(parse_local_date_time(input, offset)?).ok()
}

fn parse_offset(zone: &str) -> Option<Offset> {
  let hours = match zone.to_ascii_uppercase().as_str() {
    "Z" | "UT" | "UTC" | "GMT" => 0,
    "EDT" => -4,
    "EST" | "CDT" => -5,
    "CST" | "MDT" => -6,
    "MST" | "PDT" => -7,
    "PST" => -8,
    _ if zone.len() == 3 && zone.starts_with(['+', '-']) => zone.parse().ok()?,
    _ => {
      return ["%z", "%:z"]
        .iter()
        .find_map(|format| BrokenDownTime::parse(format, zone).ok()?.offset());
    }
  };
  Offset::from_hours(hours).ok()
}

fn parse_local_date_time(input: &str, offset: Offset) -> Option<CivilDateTime> {
  const DATE_FORMATS: &[&str] = &[
    "%Y-%m-%d",
    "%Y-%b-%d",
    "%m/%d/%y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%m-%d-%y",
    "%m-%d-%Y",
    "%m.%d.%y",
    "%m.%d.%Y",
    "%Y.%m.%d",
    "%Y.%m",
    "%Y年%m月%d日",
    "%B %d, %y",
    "%B %d, %Y",
    "%b. %d, %y",
    "%b. %d, %Y",
    "%b %d, %y",
    "%b %d, %Y",
    "%d %B %y",
    "%d %B %Y",
    "%d %b %y",
    "%d %b %Y",
    "%y%m%d",
    "%B %d",
    "%b %d",
  ];

  let today = || offset.to_datetime(Timestamp::now()).date();
  let starts_with_year = input
    .as_bytes()
    .get(..4)
    .is_some_and(|s| s.iter().all(u8::is_ascii_digit));
  for format in DATE_FORMATS {
    if format.starts_with("%Y") && !starts_with_year {
      continue;
    }
    let Ok((mut date, end)) = BrokenDownTime::parse_prefix(format, input) else {
      continue;
    };
    let rest = &input[end..];
    let time = if rest.is_empty() {
      Time::midnight()
    } else {
      if !format.ends_with('日') && !rest.starts_with(['T', 't', ',', ' ', '\t']) {
        continue;
      }
      let rest = rest.trim_start_matches(['T', 't', ',']).trim_start();
      let rest = rest.strip_prefix("at ").unwrap_or(rest);
      let Some(time) = parse_local_time(rest) else {
        continue;
      };
      time
    };
    if date.year().is_none() {
      date.set_year(Some(today().year())).ok()?;
    } else if (format.ends_with("%y") || *format == "%y%m%d") && date.year() == Some(1969) {
      // Scallop's two-digit years historically used 1970 as the cutoff.
      date.set_year(Some(2069)).ok()?;
    }
    if date.day().is_none() {
      date.set_day(Some(today().day())).ok()?;
    }
    if let Ok(date) = date.to_date() {
      return Some(date.to_datetime(time));
    }
  }
  parse_local_time(input).map(|time| today().to_datetime(time))
}

fn parse_local_time(input: &str) -> Option<Time> {
  ["%H:%M:%S%.f", "%H:%M", "%I:%M:%S%.f %p", "%I:%M %p", "%H时%M分%S秒"]
    .iter()
    .find_map(|format| Time::strptime(format, input).ok())
}

/// Format an instant using Scallop's UTC display syntax.
pub fn display_date_time(value: Timestamp) -> impl Display {
  let nanos = value.subsec_nanosecond();
  value.strftime(if nanos == 0 {
    "%F %T UTC"
  } else if nanos % 1_000_000 == 0 {
    "%F %T%.3f UTC"
  } else if nanos % 1_000 == 0 {
    "%F %T%.6f UTC"
  } else {
    "%F %T%.9f UTC"
  })
}

const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// Parse an ISO 8601 duration or Scallop's number-and-unit syntax.
pub fn parse_duration_string(input: &str) -> Option<Duration> {
  let input = input.trim();
  let parser = jiff::fmt::temporal::SpanParser::new();
  if let Some((months, elapsed)) = input.strip_prefix('R').and_then(|s| s.split_once("MO ")) {
    let months = months.parse().ok()?;
    let elapsed = parser
      .parse_duration(elapsed)
      .or_else(|_| {
        parser
          .parse_span(elapsed)?
          .to_duration(SpanRelativeTo::days_are_24_hours())
      })
      .ok()?;
    return Some(Duration::new(months, elapsed));
  }
  if let Ok(span) = parser.parse_span(input) {
    return duration_from_span(span);
  }
  if let Ok(elapsed) = parser.parse_duration(input) {
    return Some(elapsed.into());
  }
  parse_duration_terms(input)
}

fn duration_from_span(span: Span) -> Option<Duration> {
  let months = i32::from(span.get_years()) * 12 + span.get_months();
  let elapsed = span
    .try_years(0)
    .ok()?
    .try_months(0)
    .ok()?
    .to_duration(SpanRelativeTo::days_are_24_hours())
    .ok()?;
  Some(Duration::new(months, elapsed))
}

// Scallop expressions allow repeated units, exponents, and a sign on each term. See time.LICENSE.
fn parse_duration_terms(input: &str) -> Option<Duration> {
  static BARE_SECONDS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[^\w-]*(-?[0-9]+)[^\w-]*$").unwrap());
  static TERM: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
      r"(?xi)(?<integer>-?[0-9]+)(?:\.(?<fraction>[0-9]*))?(?:e(?<exponent>[-+]?[0-9]+))?[^\w]*(?<unit>[\w&&[^\d]]+)?",
    )
    .unwrap()
  });

  if let Some(seconds) = BARE_SECONDS.captures(input) {
    return Some(SignedDuration::from_secs(seconds[1].parse().ok()?).into());
  }
  let mut terms = TERM.captures_iter(input).peekable();
  terms.peek()?;
  let mut months = BigInt::ZERO;
  let mut nanos = BigInt::ZERO;
  for term in terms {
    let (calendar_months, seconds_in_nanos) = duration_unit(term.name("unit")?.as_str())?;
    let integer = term.name("integer")?.as_str();
    let fraction = term.name("fraction").map_or("", |m| m.as_str());
    let exponent = term.name("exponent");
    if calendar_months != 0 && fraction.is_empty() && exponent.is_none() {
      months += integer.parse::<BigInt>().ok()? * calendar_months;
      continue;
    }
    let mut coefficient = format!("{integer}{fraction}").parse::<BigInt>().ok()? * seconds_in_nanos;
    let exp = exponent.map_or(Some(0i64), |m| m.as_str().parse().ok())?;
    let scale = exp.saturating_sub(i64::try_from(fraction.len()).ok()?);
    if coefficient == BigInt::ZERO {
      continue;
    }
    if scale >= 0 {
      // An i64 number of seconds fits in at most 28 decimal digits of nanoseconds.
      if scale >= 28 {
        return None;
      }
      coefficient *= BigInt::from(10).pow(scale as u32);
    } else {
      let scale = scale.unsigned_abs();
      if scale > (integer.len() + fraction.len() + 17) as u64 {
        continue;
      }
      coefficient /= BigInt::from(10).pow(u32::try_from(scale).ok()?);
    }
    nanos += coefficient;
  }
  let seconds = i64::try_from(&nanos / NANOS_PER_SECOND).ok()?;
  let subsec = i32::try_from(nanos % NANOS_PER_SECOND).ok()?;
  Some(Duration::new(
    months.try_into().ok()?,
    SignedDuration::new(seconds, subsec),
  ))
}

fn duration_unit(unit: &str) -> Option<(i32, i64)> {
  let lower = unit.to_lowercase();
  let unit_is = |names: &[&str]| names.iter().any(|name| name.starts_with(&lower));
  let result = match lower.as_str() {
    s if s.starts_with('n') && unit_is(&["nanoseconds", "nsecs"]) => (0, 1),
    s if s.starts_with("mic") && unit_is(&["microseconds"]) || unit_is(&["usecs", "μsecs"]) => (0, 1_000),
    s if s.starts_with("mil") && unit_is(&["milliseconds"]) || s.starts_with("ms") && unit_is(&["msecs"]) => {
      (0, 1_000_000)
    }
    s if s.starts_with('s') && unit_is(&["seconds", "secs"]) => (0, NANOS_PER_SECOND),
    s if s.starts_with('h') && unit_is(&["hours", "hrs"]) => (0, 3_600 * NANOS_PER_SECOND),
    s if s.starts_with('d') && unit_is(&["days"]) => (0, 86_400 * NANOS_PER_SECOND),
    s if s.starts_with('w') && unit_is(&["weeks"]) => (0, 604_800 * NANOS_PER_SECOND),
    s if (s.starts_with("mo") || unit == "M") && unit_is(&["months"]) => (1, 2_629_746 * NANOS_PER_SECOND),
    s if s.starts_with('m') && unit_is(&["minutes", "mins"]) => (0, 60 * NANOS_PER_SECOND),
    s if s.starts_with('y') && unit_is(&["years", "yrs"]) => (12, 31_556_952 * NANOS_PER_SECOND),
    _ => return None,
  };
  Some(result)
}
