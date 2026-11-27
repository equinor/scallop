import unittest
from typing import assert_type

import scallopy

class TestConversion(unittest.TestCase):
  def test_convert_1(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("r", (int, int))
    ctx.add_facts("r", [(1, 3), (2, 4)])
    ctx2 = ctx.clone(scallopy.modes.top_k_proofs())
    ctx2.run()
    rows = list(ctx2.relation("r"))
    assert_type(rows[0].value, float)
    self.assertEqual(rows, [scallopy.Row((1, 3), 1.0), scallopy.Row((2, 4), 1.0)])

  def test_convert_2(self):
    ctx = scallopy.ScallopContext.from_mode(scallopy.modes.top_k_proofs())
    ctx.add_relation("r", (int, int))
    ctx.add_facts("r", [(0.5, (1, 3)), (0.8, (2, 4))])
    ctx2 = ctx.clone(scallopy.modes.unit)
    ctx2.run()
    self.assertEqual(list(ctx2.relation_tuples("r")), [(1, 3), (2, 4)])

  def test_convert_3(self):
    with self.assertWarnsRegex(DeprecationWarning, "provenance string"):
      ctx = scallopy.ScallopContext("topkproofs", k=7)
    with self.assertWarnsRegex(DeprecationWarning, "provenance string"):
      ctx2 = ctx.clone("topkproofs")
    self.assertEqual(ctx2._k, 7)

  def test_convert_4(self):
    ctx = scallopy.ScallopContext.from_mode(scallopy.modes.top_k_proofs())
    ctx2 = ctx.clone(scallopy.modes.top_k_proofs(k=5))
    self.assertEqual(ctx2._k, 5)

  def test_cloned_proof_modes_count_grouped_facts_independently(self):
    ctx = scallopy.ScallopContext.from_mode(scallopy.modes.top_k_proofs())
    ctx.add_relation("choice", int)
    ctx.add_relation("some_choice", ())
    ctx.add_rule("some_choice() = choice(x)")
    ctx.add_facts("choice", [(0.25, 1), (0.5, 2)], disjunctions=[[0, 1]])
    for mode in (scallopy.modes.top_k_proofs(), scallopy.modes.top_bottom_k_clauses()):
      cloned = ctx.clone(mode)
      cloned.run()
      self.assertEqual(list(cloned.relation("some_choice")), [scallopy.Row((), 0.625)])

  def test_loading_context_with_removed_disjunction_scoring(self):
    ctx = scallopy.ScallopContext.from_mode(scallopy.modes.top_k_proofs())
    ctx.add_relation("choice", int)
    ctx.add_facts("choice", [(0.25, 1)], disjunctions=[[0]])
    state = ctx.__getstate__()
    restored = scallopy.ScallopContext()
    with self.assertRaisesRegex(ValueError, "Legacy disjunction scoring was removed"):
      restored.__setstate__({**state, "_wmc_with_disjunctions": True})
    restored.__setstate__({**state, "_wmc_with_disjunctions": False})
    restored.run()
    self.assertEqual(list(restored.relation("choice")), [scallopy.Row((1,), 0.25)])

  def test_convert_7(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("r", (int, int))
    ctx.add_facts("r", [(1, 3), (2, 4)])
    ctx2 = ctx.clone(scallopy.modes.min_max_prob)
    ctx2.run()
    self.assertEqual(list(ctx2.relation("r")), [scallopy.Row((1, 3), 1.0), scallopy.Row((2, 4), 1.0)])


if __name__ == "__main__":
  unittest.main()
