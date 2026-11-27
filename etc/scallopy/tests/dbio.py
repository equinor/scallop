import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import scallopy

class TestIO(unittest.TestCase):
  def test_anytime_bounds_loads_weighted_csv_facts(self):
    with TemporaryDirectory() as directory:
      path = Path(directory) / "weighted.csv"
      path.write_text("0.25,1\n0.5,1\n0.75,2\ntrue,3\nfalse,4\n")
      ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
      ctx.add_relation("fact", int, load_csv=scallopy.io.CSVFileOptions(str(path), has_probability=True))
      ctx.add_rule("hit(x) = fact(x)")
      rows = ctx.evaluate_relation_bounds("hit", tolerance=0.0)
      self.assertEqual([row.tuple for row in rows], [(1,), (2,), (3,)])
      for row, expected in zip(rows, (0.625, 0.75, 1.0), strict=True):
        self.assertLessEqual(row.value.lower, expected)
        self.assertGreaterEqual(row.value.upper, expected)
        self.assertAlmostEqual(row.value.lower, expected)
        self.assertAlmostEqual(row.value.upper, expected)
      partial = ctx.evaluate_bounds("hit", (1,), max_facts=1).value
      self.assertEqual(partial.active_facts, 1)
      self.assertEqual((partial.lower, partial.upper), (0.25, 1.0))

      path.write_text("1.125,1\n")
      with self.assertRaisesRegex(ValueError, "invalid probability"):
        ctx.evaluate_bounds("hit", (1,))

  def test_anytime_bounds_session_reads_csv_only_when_started(self):
    with TemporaryDirectory() as directory:
      path = Path(directory) / "weighted.csv"
      path.write_text("0.25,1\n0.5,2\n")
      ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
      ctx.add_relation("fact", int, load_csv=scallopy.io.CSVFileOptions(str(path), has_probability=True))
      ctx.add_rule("hit() = fact(x)")
      session = ctx.start_bounds("hit", (), tolerance=0.0)
      partial = session.advance(max_facts=1).value
      path.unlink()

    resolved = session.advance(max_facts=2).value
    self.assertLessEqual(partial.lower, resolved.lower)
    self.assertLessEqual(resolved.lower, 0.625)
    self.assertGreaterEqual(resolved.upper, 0.625)
    self.assertLess(resolved.upper, partial.upper)
    self.assertEqual(resolved.stop_reason, "complete")

  def test_load_csv(self):
    ctx = scallopy.Context()
    ctx.add_relation("edge", (int, int), load_csv=scallopy.io.CSVFileOptions("core/res/testing/csv/edge.csv"))
    ctx.run()
    assert list(ctx.relation_tuples("edge")) == [(0, 1), (1, 2), (2, 3)]

  def test_load_csv_with_field(self):
    ctx = scallopy.Context()
    csv_file = scallopy.io.CSVFileOptions("core/res/testing/csv/student.csv", fields=["id", "name", "year"])
    ctx.add_relation("student", (int, str, int), load_csv=csv_file)
    ctx.run()
    assert list(ctx.relation_tuples("student")) == [(1, "alice", 2022), (2, "bob", 2023)]

  def test_load_csv_with_key_and_field(self):
    ctx = scallopy.Context()
    csv_file = scallopy.io.CSVFileOptions("core/res/testing/csv/student.csv", keys="id", fields=["name", "year"])
    ctx.add_relation("student", (int, scallopy.Symbol, str), load_csv=csv_file)
    ctx.run()
    assert list(ctx.relation_tuples("student")) == [
      (1, "name", "alice"),
      (1, "year", "2022"),
      (2, "name", "bob"),
      (2, "year", "2023"),
    ]
