use jiff::{tz::Offset, SignedDuration, Timestamp};

use scallop_core::common::duration::Duration;
use scallop_core::common::value::Value;
use scallop_core::testing::*;
use scallop_core::utils::{parse_date_time_string, parse_duration_string};

fn utc(year: i16, month: i8, day: i8) -> Timestamp {
  Offset::UTC
    .to_timestamp(jiff::civil::date(year, month, day).at(0, 0, 0, 0))
    .unwrap()
}

#[test]
fn date_type_1() {
  expect_compile(
    r#"
      type r(DateTime)
      rel r = {t"2019-01-01T00:00:00Z"}
    "#,
  )
}

#[test]
fn duration_type_1() {
  expect_compile(
    r#"
      type r(Duration)
      rel r = {d"1y5d"}
    "#,
  )
}

#[test]
fn date_1() {
  expect_interpret_result(
    r#"
      rel r = {t"2019-01-01T00:00:00Z"}
    "#,
    ("r", vec![(utc(2019, 1, 1),)]),
  )
}

#[test]
fn date_2() {
  expect_interpret_result(
    r#"
      rel r = {(0, t"2019-01-01T00:00:00Z")}
    "#,
    ("r", vec![(0, utc(2019, 1, 1))]),
  )
}

#[test]
fn bad_date_1() {
  expect_front_compile_failure(r#"rel r = {t"ABCDEF"}"#, |e| {
    e.contains("Invalid DateTime literal `ABCDEF`")
  })
}

#[test]
fn bad_duration_1() {
  expect_front_compile_failure(r#"rel r = {d"ABCDEF"}"#, |e| {
    e.contains("Invalid Duration literal `ABCDEF`")
  })
}

#[test]
fn date_plus_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-01T00:00:00Z"}
      rel q = {d"3d"}
      rel r(date + duration) = p(date) and q(duration)
    "#,
    ("r", vec![(utc(2019, 1, 4),)]),
  )
}

#[test]
fn date_minus_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel q = {d"3d"}
      rel r(date - duration) = p(date) and q(duration)
    "#,
    ("r", vec![(utc(2019, 1, 1),)]),
  )
}

#[test]
fn duration_plus_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {(d"3d", d"2d")}
      rel r(d1 + d2) = p(d1, d2)
    "#,
    ("r", vec![(Duration::days(5),)]),
  )
}

#[test]
fn relative_month_compute_1() {
  expect_interpret_result(
    r#"
      rel p = {(d"3mo", t"2019-02-01")}
      rel r(d + t) = p(d, t)
    "#,
    ("r", vec![(utc(2019, 5, 1),)]),
  )
}

#[test]
fn neg_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {(d"-3d", d"-5mo")}
    "#,
    ("p", vec![(Duration::days(-3), Duration::months(-5))]),
  )
}

#[test]
fn get_year_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel r($datetime_year(d)) = p(d)
    "#,
    ("r", vec![(2019i32,)]),
  )
}

#[test]
fn get_month_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel r($datetime_month(d)) = p(d)
    "#,
    ("r", vec![(1u32,)]),
  )
}

#[test]
fn get_month0_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel r($datetime_month0(d)) = p(d)
    "#,
    ("r", vec![(0u32,)]),
  )
}

#[test]
fn legacy_date_formats() {
  for (input, expected) in [
    ("2019-11-29 08:08:05-08", "2019-11-29T16:08:05Z"),
    ("2012-08-03 18:31:59.257000000 +0000", "2012-08-03T18:31:59.257Z"),
    ("Wed, 02 Jun 2021 06:31:39 GMT", "2021-06-02T06:31:39Z"),
    ("4/8/2014 22:05", "2014-04-08T22:05:00Z"),
    ("4-8-14 22:05", "2014-04-08T22:05:00Z"),
    ("4/8/69", "2069-04-08T00:00:00Z"),
    ("2014/04/2 03:00:51", "2014-04-02T03:00:51Z"),
    ("September 17, 2012 10:09am", "2012-09-17T10:09:00Z"),
    ("May 26, 2021, 12:49 AM PDT", "2021-05-26T07:49:00Z"),
    ("oct. 7, 1970", "1970-10-07T00:00:00Z"),
    ("oct. 7, 70", "1970-10-07T00:00:00Z"),
    ("7 oct 70", "1970-10-07T00:00:00Z"),
    ("2021-Feb-21", "2021-02-21T00:00:00Z"),
    ("171113 14:14:20", "2017-11-13T14:14:20Z"),
    ("2014年04月08日", "2014-04-08T00:00:00Z"),
    ("2014年04月08日11时25分18秒", "2014-04-08T11:25:18Z"),
    ("2021-02-21 PST", "2021-02-21T08:00:00Z"),
    ("2020-07-20+08:00", "2020-07-19T16:00:00Z"),
    ("1620021848429", "2021-05-03T06:04:08.429Z"),
  ] {
    assert_eq!(
      parse_date_time_string(input),
      Some(expected.parse().unwrap()),
      "{input}"
    );
  }
  for input in ["2021-02-29", "2021-01-01T10:00+99:00", "2014年02月30日"] {
    assert!(parse_date_time_string(input).is_none(), "{input}");
  }
}

#[test]
fn time_only_defaults_to_today_in_utc() {
  let today_before = Offset::UTC.to_datetime(Timestamp::now()).date();
  let actual = Offset::UTC.to_datetime(parse_date_time_string("4:00pm").unwrap());
  let today_after = Offset::UTC.to_datetime(Timestamp::now()).date();
  assert!(actual.date() == today_before || actual.date() == today_after);
  assert_eq!(actual.time(), jiff::civil::time(16, 0, 0, 0));
}

#[test]
fn duration_literals_preserve_units_and_precision() {
  for (input, months, seconds, nanos) in [
    ("1M1m", 1, 60, 0),
    ("1 day -15 minutes", 0, 85_500, 0),
    ("-3 days 71 hours", 0, -3_600, 0),
    ("10d1n15y", 180, 864_000, 1),
    ("1min 10 minute", 0, 660, 0),
    ("15", 0, 15, 0),
    ("16 min seconds", 0, 960, 0),
    (".:++++]][][[][15[]][seconds][]:}}}}", 0, 15, 0),
    ("1.07e-5 msecs 2usecs", 0, 0, 2_010),
    ("-0.5s", 0, 0, -500_000_000),
    ("1e-1000000000 ns", 0, 0, 0),
    ("-P1Y2M3DT4.125S", -14, -259_204, -125_000_000),
  ] {
    let expected = Duration::new(months, SignedDuration::new(seconds, nanos));
    assert_eq!(parse_duration_string(input), Some(expected), "{input}");
  }
  for input in ["1e1000000000 s", "2147483648mo", "16 17 seconds", "16 no-such-unit"] {
    assert!(parse_duration_string(input).is_none(), "{input}");
  }
}

#[test]
fn calendar_months_are_applied_before_elapsed_time() {
  expect_interpret_result(
    r#"
      rel p = {t"2020-01-31"}
      rel q = {t"2020-03-31"}
      rel plus_month(d + d"1mo") = p(d)
      rel r(a, d + d"1mo -1d", b - d"1mo -1d", a - d) = plus_month(a) and p(d) and q(b)
    "#,
    (
      "r",
      vec![(utc(2020, 2, 29), utc(2020, 2, 28), utc(2020, 3, 1), Duration::days(29))],
    ),
  )
}

#[test]
fn duration_arithmetic_preserves_calendar_and_elapsed_parts() {
  expect_interpret_result(
    r#"
      rel p = {(d"2mo -6h", d"1mo 3h")}
      rel r(a - b, b * -2, a / 2) = p(a, b)
    "#,
    (
      "r",
      vec![(
        Duration::new(1, SignedDuration::from_hours(-9)),
        Duration::new(-2, SignedDuration::from_hours(-6)),
        Duration::new(1, SignedDuration::from_hours(-3)),
      )],
    ),
  )
}

#[test]
fn equivalent_duration_literals_compare_and_deduplicate() {
  expect_interpret_result(
    r#"
      rel p = {d"1y 1d", d"12mo 24h", d"P1Y1D"}
      rel r(a) = p(a) and a > d"365d"
    "#,
    ("r", vec![(Duration::new(12, SignedDuration::from_hours(24)),)]),
  )
}

#[test]
fn temporal_display_and_duration_serialization() {
  let datetime = parse_date_time_string("2021-01-01T05:30:01.100+05:30").unwrap();
  let duration = parse_duration_string("1mo -0.5s").unwrap();
  assert_eq!(Value::DateTime(datetime).to_string(), r#"t"2021-01-01T00:00:01.1Z""#);
  assert_eq!(parse_date_time_string("2021-01-01 00:00:01.100 UTC"), Some(datetime));
  assert_eq!(parse_date_time_string(&datetime.to_string()), Some(datetime));
  assert_eq!(Value::Duration(duration).to_string(), r#"d"R1MO -PT0.5S""#);
  assert_eq!(parse_duration_string(&duration.to_string()), Some(duration));
  assert_eq!(Duration::from(SignedDuration::from_secs(90)).to_string(), "PT1M30S");
  assert_eq!(Duration::months(1).to_string(), "R1MO PT0S");
  assert_eq!(parse_duration_string("R1MO P0D"), Some(Duration::months(1)));
  for duration in [
    Duration::months(1),
    Duration::from(SignedDuration::MIN),
    Duration::from(SignedDuration::MAX),
    Duration::from(SignedDuration::from_secs(-1_000_000_000_000)),
    Duration::new(1, SignedDuration::from_secs(-1_000_000_000_000)),
  ] {
    assert_eq!(parse_duration_string(&duration.to_string()), Some(duration));
  }
  assert_eq!(
    serde_json::to_value(duration).unwrap(),
    serde_json::json!({"months": 1, "duration": 0})
  );
}

#[test]
fn datetime_fields_and_format_use_utc() {
  expect_interpret_result(
    r#"
      rel p = {t"2021-01-01T00:30:00+01:00"}
      rel r($datetime_year(d), $datetime_month(d), $datetime_day(d), $format("{}", d)) = p(d)
    "#,
    ("r", vec![(2020i32, 12u32, 31u32, "2020-12-31T23:30:00Z".to_string())]),
  )
}
