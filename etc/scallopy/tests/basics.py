import subprocess
import sys
from fractions import Fraction
from textwrap import dedent
from typing import assert_type
import unittest

import scallopy

class BasicTests(unittest.TestCase):
  def test_optional_torch_import(self):
    for missing_module in ("torch", "torch_dependency"):
      with self.subTest(missing_module=missing_module):
        result = subprocess.run(
          [sys.executable, "-c", dedent("""
            import sys

            missing_module = sys.argv[1]
            attempts = []

            class BlockTorch:
              def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch":
                  attempts.append(fullname)
                  raise ModuleNotFoundError(name=missing_module)

            sys.meta_path.insert(0, BlockTorch())
            import scallopy
            from scallopy import torch_importer

            ctx = scallopy.Context()
            ctx.add_program("rel numbers = {1, 2}")
            ctx.run()
            assert list(ctx.relation_tuples("numbers")) == [(1,), (2,)]
            typed = scallopy.Context.from_mode(scallopy.modes.unit)
            typed.add_program("rel numbers = {1, 2}")
            typed.run()
            assert list(typed.relation_rows("numbers")) == [scallopy.Row((1,), None), scallopy.Row((2,), None)]
            assert not attempts, "ordinary Scallop use tried to load Torch"

            if missing_module == "torch":
              assert not torch_importer.has_pytorch()
              try:
                scallopy.Module
              except ModuleNotFoundError as error:
                assert error.name == "torch" and "install torch" in str(error)
              else:
                raise AssertionError("Using a Torch feature should fail")
              assert len(attempts) == 1, "Torch absence wasn't cached"
            else:
              try:
                torch_importer.has_pytorch()
              except ModuleNotFoundError as error:
                assert error.name == missing_module
              else:
                raise AssertionError("A broken Torch install should surface its error")
          """), missing_module],
          capture_output=True,
          text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

  def test_native_collection_iterator_can_be_reiterated(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("numbers", int)
    ctx.add_facts("numbers", [(1,), (2,)])
    ctx.run()

    iterator = iter(ctx._internal.relation("numbers"))
    self.assertEqual(next(iterator), (1,))
    self.assertIs(iter(iterator), iterator)
    self.assertEqual(list(iterator), [(2,)])

  def test_typed_tuple_round_trip_and_invalid_arity(self):
    ctx = scallopy.Context()
    ctx.add_program("type row(number: i128, label: String)")
    fact = (2**127 - 1, "example")

    with self.assertRaisesRegex(IndexError, "Invalid tuple size"):
      ctx.add_facts("row", [(fact[0],)])

    ctx.add_facts("row", [fact])
    ctx.run()
    self.assertEqual(list(ctx.relation_tuples("row")), [fact])

  def test_typed_probability_rows_and_tuples_only_view(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.top_k_proofs(k=1))
    ctx.add_relation("weighted", (int, int))
    ctx.add_relation("deterministic", int, non_probabilistic=True)
    with self.assertRaisesRegex(ValueError, "requires tagged facts"):
      ctx.add_untagged_facts("weighted", [(1, 2)])
    ctx.add_facts("weighted", [(0.8, (1, 2)), (0.5, (1, 3))])
    ctx.add_untagged_facts("deterministic", [3])
    ctx.add_rule("result(x) = weighted(x, y), deterministic(y)")
    ctx.run()

    (row,) = ctx.relation_rows("result")
    assert_type(row.value, float)
    self.assertEqual(row, scallopy.Row((1,), 0.5))
    self.assertEqual(list(ctx.relation("result")), [row])
    self.assertEqual(list(ctx.relation_tuples("result")), [(1,)])
    (cloned_row,) = ctx.clone().relation_rows("result")
    assert_type(cloned_row.value, float)
    self.assertEqual(cloned_row, row)

  def test_proof_rows_include_proofs_and_allow_tuple_projection(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.proofs)
    ctx.add_relation("choice", int)
    ctx.add_facts("choice", [1, 2], disjunctions=[[0, 1]])
    ctx.run()

    first, second = ctx.relation_rows("choice")
    assert_type(first.value, list[list[int]])
    self.assertEqual((first.tuple, second.tuple), ((1,), (2,)))
    ((first_id,),), ((second_id,),) = first.value, second.value
    self.assertNotEqual(first_id, second_id)
    self.assertEqual(list(ctx.relation("choice")), [first, second])
    self.assertEqual(list(ctx.relation_tuples("choice")), [(1,), (2,)])

  def test_typed_debug_rows_preserve_proofs_and_autograd(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.diff_top_k_proofs_debug())
    ctx.add_relation("input", int)
    score = torch.tensor(0.7, requires_grad=True)
    ctx.add_facts("input", [((score, 1), 2)])
    ctx.add_rule("result(x) = input(x)")
    ctx.run()

    (row,) = ctx.relation_rows("result")
    result_score, proofs = row.value
    assert_type(result_score, torch.Tensor)
    assert_type(proofs, list[list[tuple[bool, int]]])
    self.assertEqual(row.tuple, (2,))
    self.assertAlmostEqual(result_score.item(), 0.7)
    self.assertEqual(proofs, [[(True, 1)]])
    result_score.backward()
    self.assertIsNotNone(score.grad)
    torch.testing.assert_close(score.grad, torch.tensor(1.0))

  def test_typed_custom_rows_keep_custom_output(self):
    class DictOutput(scallopy.ScallopProvenance):
      def recover_fn(self, internal_tag):
        return {"value": internal_tag}

    ctx = scallopy.Context.from_mode(scallopy.modes.custom(DictOutput()), no_stdlib=True)
    ctx.add_relation("input", int)
    ctx.add_facts("input", [("tag", 1)])
    ctx.run()

    self.assertEqual(list(ctx.relation_rows("input")), [scallopy.Row((1,), {"value": "tag"})])
    self.assertEqual(list(ctx.relation("input")), [scallopy.Row((1,), {"value": "tag"})])

  def test_anytime_bounds_tighten_for_recursive_overlapping_proofs(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("""
      rel 0.5::edge(0, 1)
      rel 0.5::edge(1, 2)
      rel 0.5::edge(0, 2)
      rel edge(2, 1)
      rel path(x, y) = edge(x, y) or (path(x, z) and edge(z, y))
    """)

    intervals = []
    for budget in range(4):
      row = ctx.evaluate_bounds("path", (0, 2), tolerance=0.0, max_facts=budget)
      assert_type(row.value, scallopy.ProbabilityInterval)
      self.assertEqual(row.tuple, (0, 2))
      self.assertEqual(row.value.active_facts, budget)
      self.assertLessEqual(row.value.lower, 0.625)
      self.assertGreaterEqual(row.value.upper, 0.625)
      intervals.append(row.value)

    self.assertEqual(intervals[0], scallopy.ProbabilityInterval(0.0, 1.0, 0, "fact_limit"))
    for earlier, later in zip(intervals, intervals[1:]):
      self.assertLessEqual(earlier.lower, later.lower)
      self.assertGreaterEqual(earlier.upper, later.upper)
    self.assertEqual(intervals[-1].stop_reason, "complete")
    self.assertAlmostEqual(intervals[-1].upper - intervals[-1].lower, 0.0)

    absent = ctx.evaluate_bounds("path", (1, 0), max_facts=0)
    self.assertEqual(absent.value, scallopy.ProbabilityInterval(0.0, 0.0, 0, "tolerance"))

  def test_anytime_relation_bounds_omits_zero_upper_rows(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    ctx.add_facts("choice", [(0.25, 1), (0.0, 2), (0.5, 3)], disjunctions=[[0, 1, 2]])
    ctx.add_program("""
      rel node = {1, 2, 3, 4}
      rel open(x) = node(x) and not choice(x)
      rel impossible() = choice(1) and choice(3)
    """)

    partial = ctx.evaluate_relation_bounds("choice", max_facts=2)
    self.assertEqual([row.tuple for row in partial], [(1,), (2,), (3,)])
    self.assertTrue(all(row.value.active_facts == 0 and row.value.stop_reason == "fact_limit" for row in partial))
    for relation, expected in (("choice", [(1,), (3,)]), ("open", [(1,), (2,), (3,), (4,)])):
      resolved = ctx.evaluate_relation_bounds(relation, tolerance=0.0)
      self.assertEqual([row.tuple for row in resolved], expected)
      for row in resolved:
        single = ctx.evaluate_bounds(relation, row.tuple, tolerance=0.0).value
        self.assertEqual((row.value.lower, row.value.upper), (single.lower, single.upper))
    self.assertEqual(ctx.evaluate_relation_bounds("impossible"), [])
    self.assertEqual(ctx.evaluate_bounds("impossible", ()).value.upper, 0.0)

  def test_anytime_bounds_batch_respects_query_order_and_local_inputs(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_facts("fact", [(0.5, 1)])
    ctx.add_rule("hit(x) = fact(x)")
    queries = [("hit", (2,)), ("fact", (1,)), ("hit", (2,))]
    inputs = {"fact": [[(0.75, 2)], [], [(0.125, 2)]]}
    rows = ctx.evaluate_bounds_batch(queries, inputs=inputs, tolerance=0.0)
    self.assertEqual([row.tuple for row in rows], [(2,), (1,), (2,)])
    self.assertEqual([row.value.lower for row in rows], [0.75, 0.5, 0.125])
    self.assertEqual(ctx.evaluate_bounds_batch(queries), [ctx.evaluate_bounds(*query) for query in queries])
    self.assertEqual(ctx.evaluate_bounds_batch([]), [])
    with self.assertRaisesRegex(ValueError, "Expected 3 input examples"):
      ctx.evaluate_bounds_batch(queries, inputs={"fact": [[]]})

  def test_anytime_bounds_session_keeps_its_program_and_tensor_weights(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("witness", int)
    weight = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("witness", [(weight, 1), (0.5, 2)])
    ctx.add_rule("hit() = witness(x)")
    session = ctx.start_bounds("hit", (), tolerance=0.0)
    partial = session.advance(max_facts=1).value
    self.assertEqual(partial, scallopy.ProbabilityInterval(0.25, 1.0, 1, "fact_limit"))
    self.assertEqual(session.advance(max_facts=1).value, partial)
    with self.assertRaisesRegex(ValueError, "cannot reduce"):
      session.advance(max_facts=0)
    with torch.no_grad():
      weight.fill_(0.75)
    ctx.add_facts("witness", [(1.0, 3)])
    ctx.add_program("rel alternate()\nrel hit() = alternate()")
    fresh = ctx.start_bounds("hit", ()).advance(max_facts=10).value
    self.assertEqual((fresh.lower, fresh.upper), (1.0, 1.0))
    resolved = session.advance(max_facts=2).value
    self.assertLessEqual(resolved.lower, 0.625)
    self.assertGreaterEqual(resolved.upper, 0.625)
    self.assertAlmostEqual(resolved.upper, resolved.lower)
    self.assertEqual((resolved.active_facts, resolved.stop_reason), (2, "complete"))
    self.assertEqual(session.advance(max_facts=10).value, resolved)

  def test_anytime_bounds_session_preserves_atomic_groups(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    ctx.add_facts("choice", [(0.25, 1), (0.5, 2)], disjunctions=[[0, 1]])
    ctx.add_program("""
      rel { outcome(x, true); outcome(x, false) } = choice(x)
      rel hit() = outcome(1, true)
    """)
    session = ctx.start_bounds("hit", (), tolerance=0.0)
    for budget, active in ((0, 0), (1, 0), (2, 2), (3, 2), (4, 4), (6, 6)):
      with self.subTest(budget=budget):
        row = session.advance(max_facts=budget)
        self.assertEqual(row, ctx.evaluate_bounds("hit", (), tolerance=0.0, max_facts=budget))
        self.assertEqual(row.value.active_facts, active)
        self.assertLessEqual(row.value.lower, 0.125)
        self.assertGreaterEqual(row.value.upper, 0.125)

  def test_anytime_bounds_session_timeout_returns_last_certificate(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("hit", int)
    ctx.add_facts("hit", [(0.25, 1), (0.5, 1)])
    session = ctx.start_bounds("hit", (1,), tolerance=0.0)
    fallback = session.advance(max_facts=0, timeout=0.0).value
    self.assertEqual(fallback, scallopy.ProbabilityInterval(0.0, 1.0, 0, "time_limit"))
    partial = session.advance(max_facts=1).value
    self.assertEqual(partial, scallopy.ProbabilityInterval(0.25, 1.0, 1, "fact_limit"))
    timed_out = session.advance(max_facts=2, timeout=0.0).value
    self.assertEqual(timed_out, scallopy.ProbabilityInterval(0.25, 1.0, 1, "time_limit"))
    for invalid in (-1.0, float("nan"), float("inf")):
      with self.assertRaisesRegex(ValueError, "finite, non-negative"):
        session.advance(max_facts=2, timeout=invalid)
    complete = session.advance(max_facts=2).value
    self.assertEqual(complete, ctx.evaluate_bounds("hit", (1,), tolerance=0.0, max_facts=2).value)
    self.assertEqual(session.advance(max_facts=2, timeout=0.0).value, complete)

  def test_anytime_bounds_batch_gradients_keep_example_sources(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    ctx.add_relation("gate", ())
    gate = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    p = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("gate", [(gate, ())])
    ctx.add_rule("hit() = gate() and choice(1)")
    queries = [("hit", ()), ("hit", ())]
    inputs = {"choice": [([(p, 1), (0.125, 2)], [[0, 1]]), ([(q, 1), (0.125, 2)], [[0, 1]])]}

    partial = ctx.evaluate_bounds_batch_with_gradients(queries, inputs=inputs, max_facts=1)
    self.assertTrue(all(row.value.interval.active_facts == 0 and row.value.probability is None for row in partial))
    rows = ctx.evaluate_bounds_batch_with_gradients(queries, inputs=inputs)
    for row, source in zip(rows, (p, q), strict=True):
      probability = row.value.probability
      assert probability is not None
      self.assertEqual(probability.item(), 0.5 * source.item())
      self.assertEqual(len(row.value.gradients), 2)
      self.assertIs(row.value.gradients[0].input, source)
      self.assertIs(row.value.gradients[1].input, gate)
      self.assertEqual(tuple(g.item() for g in torch.autograd.grad(probability, (source, gate))), (0.5, source.item()))

  def test_anytime_batch_shared_preparation_keeps_query_orders_and_refreshes_weights(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("witness", int)
    weight = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("witness", [(weight, 1), (0.5, 2)])
    ctx.add_rule("hit(x) = witness(x)")
    queries = [("hit", (1,)), ("witness", (2,)), ("hit", (2,)), ("hit", (1,))]
    for expected in (0.25, 0.75):
      with torch.no_grad():
        weight.fill_(expected)
      rows = ctx.evaluate_bounds_batch(queries, fact_order="required", max_facts=1, tolerance=0.0)
      self.assertEqual(rows, [
        ctx.evaluate_bounds(*query, fact_order="required", max_facts=1, tolerance=0.0) for query in queries
      ])
      self.assertEqual([row.value.lower for row in rows], [expected, 0.5, 0.5, expected])
    first, second = ctx.evaluate_bounds_batch_with_gradients([queries[0], queries[0]])
    for row in (first, second):
      probability = row.value.probability
      assert probability is not None
      self.assertEqual(probability.item(), 0.75)
      self.assertEqual(torch.autograd.grad(probability, weight)[0].item(), 1.0)

  def test_anytime_relation_gradients_retain_zero_probability_rows_with_sensitivity(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("witness", int)
    weights = tuple(torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in (0.0, 0.5))
    ctx.add_facts("witness", [(weights[0], 1), (weights[1], 2), (0.0, 3)])
    ctx.add_rule("hit(x) = witness(x)")
    self.assertEqual([row.tuple for row in ctx.evaluate_relation_bounds("hit")], [(2,)])
    partial = ctx.evaluate_relation_bounds_with_gradients("hit", max_facts=0)
    full = ctx.evaluate_relation_bounds_with_gradients("hit")
    for rows in (partial, full):
      self.assertEqual([row.tuple for row in rows], [(1,), (2,)])
      for i, row in enumerate(rows):
        expected = ctx.evaluate_bounds_with_gradients("hit", row.tuple, max_facts=row.value.interval.active_facts).value
        self.assertEqual(row.value.interval, expected.interval)
        for j, grad in enumerate(row.value.gradients):
          self.assertIs(grad.input, weights[j])
          self.assertLessEqual(grad.lower, float(i == j))
          self.assertGreaterEqual(grad.upper, float(i == j))
    self.assertTrue(all(row.value.probability is None for row in partial))
    self.assertEqual({row.value.interval.stop_reason for row in partial}, {"fact_limit"})
    zero = full[0].value.probability
    assert zero is not None
    self.assertEqual(zero.item(), 0.0)
    self.assertEqual(tuple(g.item() for g in torch.autograd.grad(zero, weights)), (1.0, 0.0))

  def test_anytime_relation_gradients_keep_normalized_categorical_boundary_rows(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    weight = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("choice", [(weight, 1), (0.5, 2)], disjunctions=[[0, 1]])
    rows = ctx.evaluate_relation_bounds_with_gradients("choice", normalize_disjunctions=True)
    self.assertEqual([row.tuple for row in rows], [(1,), (2,)])
    for row, expected in zip(rows, (2.0, -2.0), strict=True):
      (gradient,) = row.value.gradients
      self.assertLessEqual(gradient.lower, expected)
      self.assertGreaterEqual(gradient.upper, expected)
      probability = row.value.probability
      assert probability is not None
      self.assertEqual(torch.autograd.grad(probability, weight)[0].item(), expected)

  def test_anytime_bounds_distinguish_duplicate_facts_from_reused_proofs(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("irrelevant", int)
    ctx.add_facts("irrelevant", [(0.99, n) for n in range(10)])
    ctx.add_relation("witness", int)
    ctx.add_facts("witness", [(0.5, 1)])
    ctx.add_rule("hit() = witness(1)")
    ctx.add_rule("hit() = witness(1) and witness(1)")
    first = ctx.evaluate_bounds("hit", (), max_facts=1).value
    self.assertEqual(first, scallopy.ProbabilityInterval(0.5, 0.5, 1, "complete"))

    ctx.add_facts("witness", [(0.5, 1)])
    value = ctx.evaluate_bounds("hit", ()).value
    self.assertAlmostEqual(value.lower, 0.75)
    self.assertAlmostEqual(value.upper, 0.75)

  def test_anytime_bounds_prioritizes_required_facts_in_recursive_proofs(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_facts("fact", [(0.75, 1), (0.5, 2), (0.125, 0)])
    ctx.add_program("""
      rel arc = {(1, 3), (2, 3), (3, 1)}
      rel reach(x) = fact(x) and x != 0
      rel reach(x) = reach(y) and arc(y, x)
      rel hit() = fact(0) and reach(3)
      rel optional_gate() = hit() or fact(1)
      rel certain() = hit() or arc(1, 3)
    """)
    for order in ("input", "weight", "required"):
      expected_upper = 0.125 if order == "required" else 1.0
      partial = ctx.evaluate_bounds("hit", (), max_facts=1, fact_order=order).value
      self.assertEqual((partial.lower, partial.upper, partial.active_facts), (0.0, expected_upper, 1))
    complete = ctx.evaluate_bounds("hit", (), fact_order="required", tolerance=0.0).value
    self.assertAlmostEqual(complete.lower, 0.109375)
    self.assertAlmostEqual(complete.upper, 0.109375)
    alternative = ctx.evaluate_bounds("optional_gate", (), max_facts=1, fact_order="required").value
    self.assertEqual((alternative.lower, alternative.upper), (0.75, 1.0))
    certain = ctx.evaluate_bounds("certain", (), fact_order="required").value
    self.assertEqual(certain, scallopy.ProbabilityInterval(1.0, 1.0, 0, "tolerance"))
    absent = ctx.evaluate_bounds("reach", (9,), fact_order="required").value
    self.assertEqual(absent, scallopy.ProbabilityInterval(0.0, 0.0, 0, "tolerance"))

  def test_anytime_bounds_required_ranking_preserves_gradient_fact_ids(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_rule("hit() = fact(0) and fact(1)")
    p = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)
    irrelevant = torch.tensor(0.875, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("fact", [(irrelevant, 2), (p, 1), (q, 0)])
    partial = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=2, fact_order="required").value
    self.assertEqual(partial.interval.active_facts, 2)
    probability = partial.probability
    assert probability is not None
    self.assertAlmostEqual(probability.item(), 0.09375)
    self.assertEqual(tuple(g.item() for g in torch.autograd.grad(probability, (p, q, irrelevant))), (0.125, 0.75, 0.0))

  def test_anytime_bounds_lookahead_reuses_the_best_prefix(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_rule("hit() = (fact(0) and fact(1) and fact(2) and fact(4)) or fact(6)")
    p = torch.tensor(0.375, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("fact", [(0.875, 0), (0.75, 1), (0.625, 2), (0.5, 3), (p, 4), (0.25, 5), (0.125, 6)])
    base = ctx.evaluate_bounds("hit", (), max_facts=4, fact_order="required").value
    self.assertEqual((base.lower, base.upper), (0.0, 1.0))
    partial = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=4, fact_order="lookahead").value
    certain = 0.875 * 0.75 * 0.625 * 0.375
    self.assertEqual(partial.interval.active_facts, 4)
    self.assertAlmostEqual(partial.interval.lower, certain)
    self.assertEqual(partial.interval.upper, 1.0)
    (gradient,) = partial.gradients
    importance = 0.875 * 0.75 * 0.625 * (1 - 0.125)
    self.assertLessEqual(gradient.lower, importance)
    self.assertGreaterEqual(gradient.upper, importance)
    full = ctx.evaluate_bounds_with_gradients("hit", (), fact_order="lookahead").value
    probability = full.probability
    assert probability is not None
    self.assertAlmostEqual(probability.item(), 0.125 + (1 - 0.125) * certain)
    self.assertAlmostEqual(torch.autograd.grad(probability, p)[0].item(), importance)
    limited = ctx.evaluate_bounds("hit", (), fact_order="lookahead", max_facts=4).value
    self.assertEqual(limited, partial.interval)
    session = ctx.start_bounds("hit", (), fact_order="lookahead")
    session.advance(max_facts=3)
    self.assertEqual(session.advance(max_facts=4).value, limited)

  def test_anytime_bounds_adaptive_schedule_skips_only_after_a_stalled_checkpoint(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_rule("hit() = fact(1) and fact(2)")
    ctx.add_rule("rare() = fact(1) and (fact(2) or fact(3))")
    ctx.add_facts("fact", [(0.75, 0), (0.001, 1), (0.5, 2), (0.25, 3), (0.25, 4)])

    baseline = ctx.evaluate_bounds("hit", ()).value
    self.assertEqual(baseline.active_facts, 2)
    adaptive = ctx.evaluate_bounds("hit", (), checkpoint_schedule="adaptive").value
    self.assertEqual(adaptive.active_facts, 4)
    self.assertLess(adaptive.upper, baseline.upper)
    limited = ctx.evaluate_bounds("hit", (), max_facts=2, checkpoint_schedule="adaptive").value
    self.assertEqual(limited, baseline)
    required = ctx.evaluate_bounds("rare", (), fact_order="required", checkpoint_schedule="adaptive").value
    self.assertEqual(required.active_facts, 1)

  def test_anytime_bounds_enclose_nonrepresentable_probabilities(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("rel 0.1::a()\nrel 0.2::b()\nrel hit() = a() and b()")
    interval = ctx.evaluate_bounds("hit", (), tolerance=0.0).value
    exact = Fraction(0.1) * Fraction(0.2)
    self.assertLessEqual(Fraction(interval.lower), exact)
    self.assertGreaterEqual(Fraction(interval.upper), exact)

  def test_anytime_bounds_with_stratified_negation(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("""
      rel node = {1, 2}
      rel 0.25::blocked(1)
      rel 0.5::blocked(1)
      rel open(x) = node(x) and not blocked(x)
      rel closed(x) = node(x) and not open(x)
      rel both(x) = open(x) and closed(x)
      rel ok(x) = open(x) or closed(x)
    """)
    for relation, exact in [("open", 0.375), ("closed", 0.625), ("both", 0.0), ("ok", 1.0)]:
      intervals = [
        ctx.evaluate_bounds(relation, (1,), max_facts=budget, tolerance=0.0).value
        for budget in range(3)
      ]
      for interval in intervals:
        self.assertLessEqual(interval.lower, exact)
        self.assertGreaterEqual(interval.upper, exact)
      self.assertAlmostEqual(intervals[-1].lower, exact)
      self.assertAlmostEqual(intervals[-1].upper, exact)
    absent = ctx.evaluate_bounds("open", (2,), max_facts=0).value
    self.assertEqual((absent.lower, absent.upper), (1.0, 1.0))
    self.assertEqual(
      ctx.evaluate_bounds("open", (1,), max_facts=1, fact_order="required").value,
      ctx.evaluate_bounds("open", (1,), max_facts=1).value,
    )

  def test_anytime_bounds_with_gradients_encloses_partials_and_supports_autograd(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("a", int)
    ctx.add_relation("b", int)
    ctx.add_program("""
      rel node = {1}
      rel answer(x) = node(x) and (a(x) or not b(x))
    """)
    p = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("a", [(p, 1)])
    ctx.add_facts("b", [(q, 1)])
    for budget in range(3):
      result = ctx.evaluate_bounds_with_gradients("answer", (1,), max_facts=budget).value
      assert_type(result, scallopy.DifferentiableBounds)
      self.assertEqual(result.interval.active_facts, budget)
      self.assertLessEqual(result.interval.lower, 0.625)
      self.assertGreaterEqual(result.interval.upper, 0.625)
      for interval, source, expected in zip(result.gradients, (p, q), (0.5, -0.75), strict=True):
        self.assertIs(interval.input, source)
        self.assertLessEqual(interval.lower, expected)
        self.assertGreaterEqual(interval.upper, expected)
        if budget == 2:
          self.assertAlmostEqual(interval.lower, expected)
          self.assertAlmostEqual(interval.upper, expected)
      if budget < 2:
        self.assertIsNone(result.probability)
      else:
        probability = result.probability
        assert probability is not None
        self.assertEqual(probability.item(), 0.625)
        self.assertEqual(tuple(g.item() for g in torch.autograd.grad(probability, (p, q))), (0.5, -0.75))

  def test_anytime_gradient_tolerances_stop_only_when_all_requested_widths_are_met(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("fact", [(p, 0), (0.0625, 1), (q, 2), (0.5, 3), (0.5, 4)])
    ctx.add_rule("hit() = fact(0) or (fact(1) and fact(2))")

    both = ctx.evaluate_bounds_with_gradients("hit", (), gradient_tolerances=[(p, 0.1)]).value
    self.assertEqual((both.interval.stop_reason, both.interval.active_facts), ("tolerance", 2))
    self.assertIsNone(both.probability)
    self.assertGreater(both.interval.upper - both.interval.lower, 0.001)
    for bound, oracle in zip(both.gradients, (63 / 64, 1 / 32), strict=True):
      self.assertLessEqual(bound.lower, oracle)
      self.assertGreaterEqual(bound.upper, oracle)
    self.assertLessEqual(both.gradients[0].upper - both.gradients[0].lower, 0.1)

    for extra in ([(q, 0.01)], None):
      with self.subTest(extra=extra is not None):
        targets = [(p, 0.1)] + (extra or [])
        result = ctx.evaluate_bounds_with_gradients(
          "hit", (), gradient_tolerances=targets, probability_tolerance=0.001 if extra is None else None,
        ).value
        self.assertEqual((result.interval.stop_reason, result.interval.active_facts), ("tolerance", 4))
        self.assertIsNotNone(result.probability)
        self.assertLessEqual(result.interval.upper - result.interval.lower, 0.001)

    capped = ctx.evaluate_bounds_with_gradients("hit", (), gradient_tolerances=[(p, 0.1)], max_facts=1).value
    self.assertEqual((capped.interval.stop_reason, capped.interval.active_facts), ("fact_limit", 1))
    self.assertLessEqual(capped.gradients[0].lower, 63 / 64)
    self.assertGreaterEqual(capped.gradients[0].upper, 63 / 64)
    probability_only = ctx.evaluate_bounds_with_gradients("hit", (), probability_tolerance=1.0).value
    self.assertEqual((probability_only.interval.stop_reason, probability_only.interval.active_facts), ("tolerance", 0))
    self.assertEqual(len(probability_only.gradients), 2)

  def test_anytime_gradient_tolerances_match_tensor_identity_and_validate_targets(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("irrelevant", int)
    ctx.add_relation("witness", int)
    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("irrelevant", [(p, 1)])
    ctx.add_facts("witness", [(p, 1)])
    result = ctx.evaluate_bounds_with_gradients("witness", (1,), gradient_tolerances=[(p, 0.0)], max_facts=0).value
    self.assertEqual(result.interval.stop_reason, "fact_limit")
    self.assertEqual((result.gradients[0].lower, result.gradients[0].upper), (0.0, 0.0))
    self.assertGreater(result.gradients[1].upper - result.gradients[1].lower, 0.0)
    for targets, message in (
      ([(p.clone(), 0.1)], "not a trainable fact weight"),
      ([(p, -0.1)], "finite and non-negative"),
      ([(p, float("nan"))], "finite and non-negative"),
      ([(p, 0.1), (p, 0.2)], "more than one"),
      ([], "at least one"),
    ):
      with self.assertRaisesRegex(ValueError, message):
        ctx.evaluate_bounds_with_gradients("witness", (1,), gradient_tolerances=targets)

  def test_anytime_approximate_backward_certifies_coefficients_in_source_dtype(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    p = torch.tensor(0.25, dtype=torch.float16, requires_grad=True)
    q = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("fact", [(0.1, 0), (p, 1), (q, 2)])
    ctx.add_rule("hit() = fact(0) and (fact(1) or fact(2))")
    bounds = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=2).value
    self.assertIsNone(bounds.probability)
    approx = bounds.approximate_backward()
    gate = Fraction.from_float(0.1)
    delta = abs(Fraction.from_float(approx.estimate.item()) - gate * Fraction(5, 8))
    self.assertLessEqual(delta, approx.absolute_error)
    self.assertNotEqual(approx.gradients[0].estimate, 0.05)
    for coefficient, returned, exact in zip(
      approx.gradients, torch.autograd.grad(approx.estimate, (p, q)), (gate / 2, gate * Fraction(3, 4)), strict=True,
    ):
      self.assertEqual(returned.item(), coefficient.estimate)
      self.assertLessEqual(abs(Fraction.from_float(coefficient.estimate) - exact), coefficient.absolute_error)

  def test_anytime_double_backward_composes_with_shared_weights_and_nonlinear_loss(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    logit = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    weight = logit.sigmoid()
    q = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("fact", [(weight, 0), (weight, 1), (q, 2)])
    ctx.add_rule("hit() = (fact(0) and fact(1)) or fact(2)")
    probability = ctx.evaluate_bounds_with_gradients("hit", (), second_order=True).value.probability
    assert probability is not None
    expected = weight.square() * (1 - q) + q
    self.assertEqual(probability.item(), expected.item())
    inputs = (logit, q)
    actual = torch.autograd.grad(probability.square(), inputs, create_graph=True)
    oracle = torch.autograd.grad(expected.square(), inputs, create_graph=True)
    torch.testing.assert_close(actual, oracle, atol=1e-14, rtol=0)
    for row, expected_row in zip(actual, oracle, strict=True):
      torch.testing.assert_close(
        torch.autograd.grad(row, inputs, retain_graph=True),
        torch.autograd.grad(expected_row, inputs, retain_graph=True),
        atol=1e-14, rtol=0,
      )

  def test_anytime_double_backward_matches_categorical_possible_worlds(self):
    from itertools import product

    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    weights = tuple(
      torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in (0.0, 0.5, 0.25, 0.25, 0.5, 0.25)
    )
    ctx.add_facts("fact", [(weight, i) for i, weight in enumerate(weights)], disjunctions=[[0, 1, 5], [2, 3]])
    ctx.add_rule("hit() = (fact(0) and fact(2)) or (fact(1) and not fact(3)) or (fact(4) and fact(3))")
    for normalize in (False, True):
      with self.subTest(normalize=normalize):
        p, q, r, s, gate, unused = weights
        if normalize:
          total = p + q + unused
          first = [(0, p / total), (1, q / total), (5, unused / total)]
          second = [(2, r / (r + s)), (3, s / (r + s))]
        else:
          first = [(0, p), (1, q), (5, unused), (None, 1 - p - q - unused)]
          second = [(2, r), (3, s), (None, 1 - r - s)]
        worlds = product(first, second, ((False, 1 - gate), (True, gate)))
        expected = torch.stack([
          a * b * c for (x, a), (y, b), (z, c) in worlds
          if (x == 0 and y == 2) or (x == 1 and y != 3) or (z and y == 3)
        ]).sum()
        unresolved = ctx.evaluate_bounds_with_gradients(
          "hit", (), max_facts=1, second_order=True, normalize_disjunctions=normalize,
        ).value
        self.assertIsNone(unresolved.probability)
        probability = ctx.evaluate_bounds_with_gradients(
          "hit", (), second_order=True, normalize_disjunctions=normalize,
        ).value.probability
        assert probability is not None
        torch.testing.assert_close(probability, expected, atol=1e-14, rtol=0)
        actual = torch.autograd.grad(probability, weights, create_graph=True)
        oracle = torch.autograd.grad(expected, weights, create_graph=True)
        torch.testing.assert_close(actual, oracle, atol=1e-14, rtol=0)
        for row, expected_row in zip(actual, oracle, strict=True):
          torch.testing.assert_close(
            torch.autograd.grad(row, weights, retain_graph=True),
            torch.autograd.grad(expected_row, weights, retain_graph=True, materialize_grads=True),
            atol=1e-14, rtol=0,
          )

  def test_anytime_double_backward_keeps_rare_effects_under_negation(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    rare = 2.0 ** -60
    ctx.add_facts("fact", [(p, 0), (q, 1), (rare, 2)])
    ctx.add_program("rel rare() = fact(0) and fact(1) and fact(2)\nrel hit() = not rare()")
    probability = ctx.evaluate_bounds_with_gradients("hit", (), second_order=True).value.probability
    assert probability is not None
    (first,) = torch.autograd.grad(probability, p, create_graph=True)
    self.assertEqual(first.item(), -0.5 * rare)
    (mixed,) = torch.autograd.grad(first, q)
    self.assertEqual(mixed.item(), -rare)

  def test_anytime_bounds_with_gradients_tracks_boundary_weights_and_shared_tensors(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_rule("hit() = fact(1) or fact(2)")
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    weight = 2 * p
    ctx.add_facts("fact", [(weight, 1), (weight, 2)])
    result = ctx.evaluate_bounds_with_gradients("hit", ()).value
    self.assertEqual(result.interval.active_facts, 2)
    self.assertEqual(len(result.gradients), 2)
    probability = result.probability
    assert probability is not None
    self.assertEqual(probability.item(), 0.0)
    probability.backward()
    gradient = p.grad
    assert gradient is not None
    self.assertEqual(gradient.item(), 4.0)

  def test_anytime_bounds_with_gradients_covers_facts_hidden_by_approximation(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_rule("hit() = (fact(1) and fact(2)) or fact(3)")
    weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("fact", [(weight, 1), (0.5, 2), (0.5, 3)])
    result = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=1).value
    self.assertEqual(result.interval.active_facts, 1)
    self.assertIsNone(result.probability)
    (gradient,) = result.gradients
    self.assertLessEqual(gradient.lower, 0.25)
    self.assertGreaterEqual(gradient.upper, 0.25)

  def test_anytime_bounds_batch_gradients_cover_different_endpoint_supports(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    ctx.add_rule("hit() = fact(i) and i < 8 and not fact(8)")
    values = (0.0, *([0.125] * 7), 0.375)
    weights = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in values)
    ctx.add_facts("fact", [(weight, i) for i, weight in enumerate(weights)])
    missed = 0.875 ** 7
    expected = ((1 - values[8]) * missed, *([(1 - values[8]) * 0.875 ** 6] * 7), missed - 1)
    for budget in (8, 9):
      result = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=budget).value
      self.assertEqual(result.interval.active_facts, budget)
      for bound, derivative in zip(result.gradients, expected, strict=True):
        self.assertLessEqual(bound.lower, derivative)
        self.assertGreaterEqual(bound.upper, derivative)
      if budget == 8:
        self.assertIsNone(result.probability)
      else:
        probability = result.probability
        assert probability is not None
        self.assertEqual(tuple(g.item() for g in torch.autograd.grad(probability, weights)), expected)

  def test_anytime_grouped_count_distinguishes_absent_and_empty_groups(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("seed", int)
    ctx.add_relation("domain", int)
    ctx.add_facts("seed", [(0.25, 1)])
    ctx.add_facts("domain", [(0.5, 1), (1.0, 2)])
    ctx.add_program("""
      rel member(1, 1) = seed(1)
      rel member(2, 1) = seed(1)
      rel implicit(g, n) = n := count(x: member(x, g))
      rel explicit(g, n) = n := count(x: member(x, g) where g: domain(g))
      rel empty(g, n) = n := count(x: member(x, g) and x == 3 where g: domain(g))
      rel absent() = not implicit(1, 0)
    """)

    for relation, query, expected in (
      ("implicit", (1, 0), 0.0), ("implicit", (1, 1), 0.0), ("implicit", (1, 2), 0.25),
      ("explicit", (1, 0), 0.375), ("explicit", (1, 2), 0.125), ("explicit", (2, 0), 1.0),
      ("empty", (1, 0), 0.5), ("empty", (2, 0), 1.0), ("absent", (), 1.0),
    ):
      partial = ctx.evaluate_bounds(relation, query, tolerance=0.0, max_facts=0).value
      self.assertLessEqual(partial.lower, expected)
      self.assertGreaterEqual(partial.upper, expected)
      full = ctx.evaluate_bounds(relation, query, tolerance=0.0).value
      self.assertLessEqual(full.lower, expected)
      self.assertGreaterEqual(full.upper, expected)
      self.assertAlmostEqual(full.lower, expected)
      self.assertAlmostEqual(full.upper, expected)

  def test_anytime_grouped_count_uses_correlated_domain_and_membership(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("seed", int)
    weight = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("seed", [(weight, 1)])
    ctx.add_program("""
      rel member(1, 1) = seed(1)
      rel domain(1) = seed(1)
      rel result(g, n) = n := count(x: member(x, g) where g: domain(g))
    """)
    self.assertEqual([row.tuple for row in ctx.evaluate_relation_bounds("result")], [(1, 1)])
    result = ctx.evaluate_bounds_with_gradients("result", (1, 1)).value
    self.assertEqual(result.interval.lower, 0.25)
    self.assertEqual(result.interval.upper, 0.25)
    self.assertAlmostEqual(result.gradients[0].lower, 1.0)
    self.assertAlmostEqual(result.gradients[0].upper, 1.0)
    probability = result.probability
    assert probability is not None
    self.assertEqual(torch.autograd.grad(probability, weight)[0].item(), 1.0)

  def test_anytime_extrema_and_exists_follow_possible_worlds(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("item", (int, int, int))
    ctx.add_facts("item", [(0.25, (1, 10, 4)), (0.5, (1, 20, 2)), (0.5, (1, 30, 2))])
    ctx.add_program("""
      rel smallest(g, v) = v := min(v: item(g, _, v))
      rel largest(g, v) = v := max(v: item(g, _, v))
      rel winners(g, i) = i := argmin[i](v: item(g, i, v))
      rel nonempty(g, b) = b := exists(i: item(g, i, _))
      rel found(b) = b := exists(i: item(_, i, _))
    """)
    for relation, query, expected in (
      ("smallest", (1, 2), 0.75), ("smallest", (1, 4), 0.0625),
      ("largest", (1, 2), 0.5625), ("largest", (1, 4), 0.25),
      ("winners", (1, 10), 0.0625), ("winners", (1, 20), 0.5), ("winners", (1, 30), 0.5),
      ("nonempty", (1, True), 0.8125), ("nonempty", (1, False), 0.0),
      ("found", (False,), 0.1875),
    ):
      partial = ctx.evaluate_bounds(relation, query, max_facts=1).value
      self.assertLessEqual(partial.lower, expected)
      self.assertGreaterEqual(partial.upper, expected)
      full = ctx.evaluate_bounds(relation, query, tolerance=0.0).value
      self.assertLessEqual(full.lower, expected)
      self.assertGreaterEqual(full.upper, expected)
      self.assertAlmostEqual(full.lower, expected)
      self.assertAlmostEqual(full.upper, expected)

  def test_anytime_bounds_counts_distinct_tuples_in_the_same_world(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("seed", int)
    ctx.add_program("""
      rel member(1) = seed(1)
      rel member(2) = seed(1)
      rel member(3) = seed(2)
      rel result(n) = n := count(x: member(x))
      rel enough() = result(n) and n >= 2
    """)
    empty = ctx.evaluate_bounds("result", (0,)).value
    self.assertEqual((empty.lower, empty.upper), (1.0, 1.0))
    weight = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("seed", [(weight, 1), (0.5, 2)])
    for budget in range(3):
      for count, exact in enumerate((0.375, 0.375, 0.125, 0.125)):
        value = ctx.evaluate_bounds("result", (count,), max_facts=budget, tolerance=0.0).value
        self.assertLessEqual(value.lower, exact)
        self.assertGreaterEqual(value.upper, exact)
        if budget == 2:
          self.assertAlmostEqual(value.lower, exact)
          self.assertAlmostEqual(value.upper, exact)
    result = ctx.evaluate_bounds_with_gradients("enough", ()).value
    probability = result.probability
    assert probability is not None
    self.assertEqual(probability.item(), 0.25)
    self.assertEqual(torch.autograd.grad(probability, weight)[0].item(), 1.0)

  def test_anytime_numeric_aggregates_merge_worlds_without_inventing_implicit_groups(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("seed", int)
    ctx.add_facts("seed", [(0.25, 1), (0.5, 2), (0.5, 3)])
    ctx.add_program("""
      rel member(1, 10, 0) = seed(1)
      rel member(1, 20, 2) = seed(1)
      rel member(1, 30, 2) = seed(2)
      rel member(3, 40, 0) = seed(3)
      rel domain(1) = seed(2)
      rel domain(2)
      rel total(g, s) = s := sum[i](v: member(g, i, v))
      rel product(g, p) = p := prod[i](v: member(g, i, v) where g: domain(g))
    """)
    for relation, query, expected in (
      ("total", (1, 0), 0.0), ("total", (1, 2), 0.5), ("total", (1, 4), 0.125),
      ("total", (3, 0), 0.5), ("product", (1, 0), 0.125), ("product", (1, 2), 0.375),
      ("product", (1, 1), 0.0), ("product", (2, 1), 1.0),
    ):
      for budget in (0, 1, 3):
        with self.subTest(relation=relation, query=query, budget=budget):
          bound = ctx.evaluate_bounds(relation, query, tolerance=0.0, max_facts=budget).value
          self.assertLessEqual(bound.lower, expected)
          self.assertGreaterEqual(bound.upper, expected)
          if budget == 3:
            self.assertAlmostEqual(bound.lower, expected)
            self.assertAlmostEqual(bound.upper, expected)

  def test_anytime_numeric_aggregates_wrap_integers_and_fold_floats(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("small", "i8")
    ctx.add_facts("small", [(1.0, 120), (0.5, 2), (1.0, 6)])
    ctx.add_relation("decimal", "f64")
    ctx.add_facts("decimal", [(1.0, 1.5), (0.25, 2.5)])
    ctx.add_relation("ordered", (int, "f64"))
    ctx.add_facts("ordered", [(1.0, (3, 1.0)), (1.0, (2, -1e16)), (0.5, (1, 1e16))])
    ctx.add_program("""
      rel total(s) = s := sum(x: small(x))
      rel product(p) = p := prod(x: small(x))
      rel fraction(p) = p := prod(x: decimal(x))
      rel ordered_sum(s) = s := sum[i](x: ordered(i, x))
      rel empty_sum(s) = s := sum(x: small(x) and x > 120)
    """)
    for relation, query, expected in (
      ("total", (-128,), 0.5), ("product", (-96,), 0.5), ("fraction", (3.75,), 0.25),
      ("ordered_sum", (1.0,), 0.5), ("empty_sum", (0,), 1.0),
    ):
      with self.subTest(relation=relation):
        value = ctx.evaluate_bounds(relation, query, tolerance=0.0).value
        self.assertEqual((value.lower, value.upper), (expected, expected))

  def test_anytime_forall_over_an_explicit_domain_counts_counterexamples(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("has_color", int)
    ctx.add_facts("has_color", [(0.25, 1), (0.5, 2)])
    ctx.add_program("""
      rel group = {1, 2}
      rel member = {(1, 1), (1, 2)}
      rel all_colored(g, b) = b := forall(x: member(g, x) => has_color(x) where g: group(g))
    """)
    for query, expected in (((1, True), 0.125), ((1, False), 0.875), ((2, True), 1.0)):
      value = ctx.evaluate_bounds("all_colored", query, tolerance=0.0).value
      self.assertLessEqual(value.lower, expected)
      self.assertGreaterEqual(value.upper, expected)
      self.assertAlmostEqual(value.lower, expected)
      self.assertAlmostEqual(value.upper, expected)

  def test_anytime_computed_projections_merge_proofs_for_the_same_output(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("item", ("i8", "i8"))
    ctx.add_facts("item", [(0.25, (2, 5)), (0.5, (3, 4)), (0.25, (127, 1))])
    ctx.add_program("""
      rel total(s) = item(a, b) and s == a + b
      rel bucket(if s < 10 then s - 1 else s) = total(s)
    """)
    for query, expected in (((6,), 0.625), ((127,), 0.25), ((7,), 0.0)):
      partial = ctx.evaluate_bounds("bucket", query, max_facts=1).value
      self.assertLessEqual(partial.lower, expected)
      self.assertGreaterEqual(partial.upper, expected)
      full = ctx.evaluate_bounds("bucket", query, tolerance=0.0).value
      self.assertLessEqual(full.lower, expected)
      self.assertGreaterEqual(full.upper, expected)
      self.assertAlmostEqual(full.lower, expected)
      self.assertAlmostEqual(full.upper, expected)

  def test_anytime_rejects_recursive_projections_and_unchecked_arithmetic(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("""
      rel start = {1}
      rel chain(x) = start(x)
      rel chain(y) = chain(x) and y == x + 1
      rel ratio(y) = start(x) and y == x / (x - 1)
      rel unsupported_avg(a) = a := avg(x: start(x))
      rel event = {t"2026-01-01T00:00:00Z"}
      rel shifted(t + d"1d") = event(t)
    """)
    for relation, message in (
      ("chain", "computed projections in recursive strata"),
      ("ratio", "arithmetic operator"),
      ("unsupported_avg", "aggregation outside"),
      ("shifted", "arithmetic operator"),
    ):
      with self.subTest(relation=relation), self.assertRaisesRegex(ValueError, message):
        ctx.evaluate_relation_bounds(relation)

  def test_anytime_deterministic_foreign_calls_preserve_fact_membership(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("source", (str, int))
    ctx.add_facts("source", [(0.25, ("  A,B ", 2)), (0.5, ("a,b", 3))])
    ctx.add_relation("reading", "f64")
    ctx.add_facts("reading", [(0.5, 1.0), (0.25, 0.0), (0.5, -1.0)])
    ctx.add_program("""
      rel tokens(word) = source(text, _) and normalized == $string_lower($string_trim(text))
                         and string_split(normalized, ",", word)
      rel number(i) = source(_, end) and range<i32>(0, end, i)
      rel constant(i) = range<i32>(5, 7, i)
      rel log_value($log(value)) = reading(value)
      rel close() = reading(value) and float_eq<f64>(value, 1.0001)
      rel missing() = not tokens("a")
    """)
    for relation, query, expected in (
      ("tokens", ("a",), 0.625), ("number", (1,), 0.625), ("number", (2,), 0.5),
      ("constant", (5,), 1.0), ("constant", (7,), 0.0),
      ("log_value", (0.0,), 0.5), ("close", (), 0.5), ("missing", (), 0.375),
    ):
      for budget in (0, 1, 3):
        with self.subTest(relation=relation, query=query, budget=budget):
          value = ctx.evaluate_bounds(
            relation, query, max_facts=budget, tolerance=0.0, fact_order="required",
          ).value
          self.assertLessEqual(value.lower, expected)
          self.assertGreaterEqual(value.upper, expected)
          if budget == 3:
            self.assertAlmostEqual(value.lower, expected)
            self.assertAlmostEqual(value.upper, expected)

  def test_anytime_rejects_impure_tagged_and_recursive_generating_calls(self):
    called = []

    @scallopy.foreign_function
    def side_effect(value: str) -> str:
      called.append(value)
      return value

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.register_foreign_function(side_effect)
    ctx.add_program("""
      rel input = {1}
      rel callback($side_effect("x")) = input(1)
      rel scored() = soft_eq<f64>(1.0, 2.0)
      rel growing(x) = input(x)
      rel growing(y) = growing(x) and range<i32>(0, x, y)
    """)
    for relation, message in (
      ("callback", "foreign function `side_effect`"),
      ("scored", "foreign predicate `soft_eq"),
      ("growing", "value-generating predicates.*recursive"),
    ):
      with self.subTest(relation=relation), self.assertRaisesRegex(ValueError, message):
        ctx.evaluate_relation_bounds(relation, max_facts=0)
    self.assertEqual(called, [])

  def test_anytime_goal_annotation_still_evaluates_all_proofs(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(0.25, (0, 2)), (0.5, (0, 1)), (1.0, (1, 2))])
    ctx.add_program("""
      rel reachable(0)
      @scheduler("lfp")
      rel reachable(y) = reachable(x) and edge(x, y)
      @goal
      rel answer() = reachable(2)
    """)
    bound = ctx.evaluate_bounds("answer", (), tolerance=0.0).value
    self.assertLessEqual(bound.lower, 0.625)
    self.assertGreaterEqual(bound.upper, 0.625)
    self.assertAlmostEqual(bound.lower, 0.625)
    self.assertAlmostEqual(bound.upper, 0.625)

  def test_anytime_rejects_non_lfp_scheduling_and_indexed_storage(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("""
      @storage("indexed_vec")
      type indexed(usize, i32)
      rel indexed = {(0, 7), (0, 8)}
      @scheduler("beam")
      type beam(i32)
      rel beam = {1}
    """)
    for relation, message in (("indexed", "non-sorted storage"), ("beam", "non-LFP scheduler")):
      with self.subTest(relation=relation), self.assertRaisesRegex(ValueError, message):
        ctx.evaluate_relation_bounds(relation)

  def test_anytime_rule_choice_is_shared_across_body_branches(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("left", int)
    ctx.add_relation("right", int)
    ctx.add_facts("left", [(0.5, 1)])
    ctx.add_facts("right", [(0.5, 1)])
    ctx.add_program("""
      rel { face(x, 1); face(x, 2); face(x, 3) } = left(x) or right(x)
      rel { fixed(x, true); fixed(x, false) } = x == 1 or left(x)
      rel conflict() = face(1, 1) and face(1, 2)
      rel absent() = not face(1, 1)
    """)
    for relation, query, expected in (
      ("face", (1, 1), 0.25), ("fixed", (1, True), 0.5),
      ("conflict", (), 0.0), ("absent", (), 0.75),
    ):
      for budget in (0, 2, 4, 5):
        with self.subTest(relation=relation, budget=budget):
          bound = ctx.evaluate_bounds(relation, query, max_facts=budget, tolerance=0.0).value
          self.assertLessEqual(bound.lower, expected)
          self.assertGreaterEqual(bound.upper, expected)
          if budget == 5:
            self.assertAlmostEqual(bound.lower, expected)
            self.assertAlmostEqual(bound.upper, expected)
    capped = ctx.evaluate_bounds("face", (1, 1), max_facts=4, tolerance=0.0).value
    self.assertEqual((capped.active_facts, capped.stop_reason), (2, "fact_limit"))
    rows = ctx.evaluate_relation_bounds("face", max_facts=5, tolerance=0.0, normalize_disjunctions=True)
    self.assertEqual({row.tuple for row in rows}, {(1, 1), (1, 2), (1, 3)})
    for row in rows:
      bound = row.value
      self.assertLessEqual(bound.lower, 0.25)
      self.assertGreaterEqual(bound.upper, 0.25)
      self.assertAlmostEqual(bound.lower, bound.upper)

  def test_anytime_rule_choices_are_independent_per_rule_and_grounding(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("""
      rel source = {1, 2}
      rel other(1)
      rel { flag(x, true); flag(x, false) } = source(x)
      rel { flag(x, true); flag(x, false) } = other(x)
      rel { repeated(x, 0); repeated(x, 0); repeated(x, 1) } = source(x)
      rel { certain(x, 0); certain(x, 0) } = source(x)
      rel { fallback(x, true); fallback(x, false) } = source(x) and not flag(x, true)
      rel joint() = flag(1, true) and flag(2, false)
      rel conflicting(x) = flag(x, true) and flag(x, false)
    """)
    for relation, query, expected in (
      ("flag", (1, True), 0.75), ("flag", (2, True), 0.5), ("joint", (), 0.375),
      ("conflicting", (1,), 0.5), ("conflicting", (2,), 0.0), ("repeated", (1, 1), 0.5),
      ("fallback", (1, True), 0.125),
    ):
      value = ctx.evaluate_bounds(relation, query, max_facts=12, tolerance=0.0).value
      self.assertLessEqual(value.lower, expected)
      self.assertGreaterEqual(value.upper, expected)
      self.assertAlmostEqual(value.lower, expected)
      self.assertAlmostEqual(value.upper, expected)
    certain = ctx.evaluate_bounds("certain", (1, 0), max_facts=0).value
    self.assertEqual((certain.lower, certain.upper, certain.active_facts), (1.0, 1.0, 0))

  def test_anytime_rule_choice_rejects_specialized_different_head_patterns(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("rel { outcome(x, 0); outcome(y, 1) } = x == 0 and y == 1")
    with self.assertRaisesRegex(ValueError, "shared source head variable pattern"):
      ctx.evaluate_relation_bounds("outcome")

  def test_anytime_rule_choices_survive_recursion_and_keep_input_gradients(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("start", int)
    weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("start", [(weight, 0)])
    ctx.add_program("""
      rel edge = {(0, 1), (1, 0), (1, 2)}
      rel reachable(x) = start(x)
      rel { step(x, true); step(x, false) } = reachable(x)
      rel reachable(y) = step(x, true) and edge(x, y)
      rel failed() = not reachable(2)
    """)
    for budget in (0, 1, 2, 3, 5, 7):
      value = ctx.evaluate_bounds_with_gradients("failed", (), max_facts=budget).value
      self.assertLessEqual(value.interval.lower, 0.875)
      self.assertGreaterEqual(value.interval.upper, 0.875)
      (gradient,) = value.gradients
      self.assertLessEqual(gradient.lower, -0.25)
      self.assertGreaterEqual(gradient.upper, -0.25)
      if budget == 7:
        self.assertAlmostEqual(value.interval.lower, 0.875)
        self.assertAlmostEqual(value.interval.upper, 0.875)
        probability = value.probability
        assert probability is not None
        self.assertEqual(torch.autograd.grad(probability, weight)[0].item(), -0.25)

  def test_anytime_joint_choice_preserves_correlation_across_relations_and_budgets(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    for relation in ("left", "right", "ordinary"):
      ctx.add_relation(relation, int)
    ctx.add_facts("ordinary", [(0.5, 0), (0.5, 1)], disjunctions=[[0, 1]])
    ctx.add_joint_choice([
      (0.25, {"left": [0, 0, 1], "right": [0]}),
      (0.5, {"right": [1]}),
      (0.125, {}),
    ])
    ctx.add_program("""
      rel together() = left(0) and right(0)
      rel disagreement() = left(0) and not right(0)
      rel mixed() = left(0) and right(1)
      rel independent() = left(0) and ordinary(0)
      rel no_right() = not right(0) and not right(1)
      rel pair() = n := count(x: left(x)), n == 2
    """)
    partial = ctx.start_bounds("together", (), tolerance=0.0)
    self.assertEqual(partial.advance(max_facts=2).value, scallopy.ProbabilityInterval(0.0, 1.0, 0, "fact_limit"))
    for relation, expected in (
      ("together", 0.25), ("disagreement", 0.0), ("mixed", 0.0),
      ("independent", 0.125), ("no_right", 0.25), ("pair", 0.25),
    ):
      with self.subTest(relation=relation):
        bound = ctx.evaluate_bounds(relation, (), tolerance=0.0).value
        self.assertLessEqual(bound.lower, expected)
        self.assertGreaterEqual(bound.upper, expected)
        self.assertAlmostEqual(bound.lower, bound.upper)
    self.assertEqual(partial.advance(max_facts=3).value, ctx.evaluate_bounds("together", (), tolerance=0.0).value)
    cloned = ctx.clone()
    self.assertEqual(cloned.evaluate_relation_bounds("left"), ctx.evaluate_relation_bounds("left"))

  def test_anytime_joint_choice_normalization_keeps_sliced_and_empty_outcomes(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("left", int)
    ctx.add_relation("right", int)
    p, q, r = (torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in (0.25, 0.5, 0.25))
    ctx.add_joint_choice([(p, {"left": [1, 2], "right": [1]}), (q, {"right": [2]}), (r, {})])
    ctx.add_rule("both() = left(1) and left(2)")
    rows = ctx.evaluate_bounds_batch_with_gradients(
      [("left", (1,)), ("both", ())], normalize_disjunctions=True,
    )
    for row in rows:
      self.assertEqual(len(row.value.gradients), 3)
      probability = row.value.probability
      assert probability is not None
      self.assertEqual(probability.item(), 0.25)
      slopes = torch.autograd.grad(probability, (p, q, r))
      self.assertEqual(tuple(slope.item() for slope in slopes), (0.75, -0.25, -0.25))
      for interval, slope in zip(row.value.gradients, slopes, strict=True):
        self.assertLessEqual(interval.lower, slope.item())
        self.assertGreaterEqual(interval.upper, slope.item())
    partial = ctx.evaluate_bounds_with_gradients(
      "both", (), normalize_disjunctions=True, max_facts=2, gradient_tolerances=[(r, 0.01)],
    ).value
    self.assertEqual((partial.interval.stop_reason, partial.interval.active_facts), ("fact_limit", 0))
    resolved = ctx.evaluate_bounds_with_gradients(
      "both", (), normalize_disjunctions=True, max_facts=3, gradient_tolerances=[(r, 0.01)], second_order=True,
    ).value
    self.assertEqual((resolved.interval.stop_reason, resolved.interval.active_facts), ("complete", 3))
    self.assertLessEqual(resolved.gradients[2].upper - resolved.gradients[2].lower, 0.01)
    probability = resolved.probability
    assert probability is not None
    (first,) = torch.autograd.grad(probability, r, create_graph=True)
    self.assertEqual(torch.autograd.grad(first, r)[0].item(), 0.5)

  def test_anytime_joint_choice_rejects_invalid_weights_and_empty_choices(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("fact", int)
    with self.assertRaisesRegex(ValueError, "at least one fact"):
      ctx.add_joint_choice([(0.25, {})])
    with self.assertRaisesRegex(ValueError, "invalid probability"):
      ctx.add_joint_choice([(1.5, {"fact": [1]})])
    ctx.add_joint_choice([(1.0, {"fact": [1]}), (1.0, {"fact": [2]})])
    with self.assertRaisesRegex(ValueError, "exceeds one"):
      ctx.evaluate_bounds("fact", (1,))
    row = ctx.evaluate_bounds("fact", (1,), tolerance=0.0, normalize_disjunctions=True)
    self.assertLessEqual(row.value.lower, 0.5)
    self.assertGreaterEqual(row.value.upper, 0.5)
    self.assertAlmostEqual(row.value.lower, row.value.upper)

  def test_anytime_bounds_exclusive_input_groups_match_possible_worlds(self):
    from itertools import product

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    ctx.add_facts("choice", [(0.25, 1), (0.5, 2), (0.5, 9)], disjunctions=[[0, 1]])
    ctx.add_program("""
      rel choice = {0.125::3; 0.875::4}
      rel impossible() = choice(1) and choice(2)
      rel exhaustive() = choice(3) or choice(4)
      rel either() = choice(1) or choice(2)
      rel none() = not either()
      rel many() = n := count(x: choice(x)), n >= 2
      rel answer() = (choice(1) and choice(3)) or (many() and not choice(4))
    """)
    first = ((1, Fraction(1, 4)), (2, Fraction(1, 2)), (None, Fraction(1, 4)))
    second = ((3, Fraction(1, 8)), (4, Fraction(7, 8)))
    independent = ((9, Fraction(1, 2)), (None, Fraction(1, 2)))
    exact = Fraction(0)
    for (a, pa), (b, pb), (c, pc) in product(first, second, independent):
      present = {a, b, c} - {None}
      if {1, 3} <= present or (len(present) >= 2 and 4 not in present):
        exact += pa * pb * pc
    for relation, expected in {
      "answer": exact, "impossible": Fraction(0), "exhaustive": Fraction(1), "none": Fraction(1, 4),
    }.items():
      previous = (0.0, 1.0)
      for budget, active in enumerate((0, 0, 2, 2, 4, 5)):
        value = ctx.evaluate_bounds(relation, (), tolerance=0.0, max_facts=budget).value
        self.assertLessEqual(Fraction(value.lower), expected)
        self.assertGreaterEqual(Fraction(value.upper), expected)
        self.assertLessEqual(previous[0], value.lower)
        self.assertGreaterEqual(previous[1], value.upper)
        self.assertLessEqual(value.active_facts, active)
        if relation == "answer":
          self.assertEqual(value.active_facts, active)
        previous = (value.lower, value.upper)
      self.assertAlmostEqual(previous[0], float(expected))
      self.assertAlmostEqual(previous[1], float(expected))

  def test_anytime_bounds_exclusive_gradients_keep_sliced_and_zero_weight_outcomes(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    ctx.add_relation("other", int)
    ctx.add_rule("hit() = choice(1)")
    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    zero = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    ctx._internal.add_facts("choice", [((p, 0), (1,)), ((zero, 0), (2,))])
    ctx._internal.add_facts("other", [((q, 0), (1,))])
    for normalize, expected_probability, expected_partials in (
      (False, Fraction(1, 2), (Fraction(1), Fraction(0), Fraction(0))),
      (True, Fraction(2, 3), (Fraction(4, 9), Fraction(-8, 9), Fraction(-8, 9))),
    ):
      expected = dict(zip(map(id, (p, q, zero)), expected_partials, strict=True))
      for budget in (0, 2, 3):
        result = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=budget, normalize_disjunctions=normalize).value
        self.assertEqual(result.interval.active_facts, 3 if budget == 3 else 0)
        self.assertLessEqual(Fraction(result.interval.lower), expected_probability)
        self.assertGreaterEqual(Fraction(result.interval.upper), expected_probability)
        for gradient in result.gradients:
          self.assertLessEqual(Fraction(gradient.lower), expected[id(gradient.input)])
          self.assertGreaterEqual(Fraction(gradient.upper), expected[id(gradient.input)])
        self.assertEqual(len(result.gradients), 3)
        if budget < 3:
          self.assertIsNone(result.probability)
        else:
          probability = result.probability
          assert probability is not None
          self.assertAlmostEqual(probability.item(), float(expected_probability))
          for actual, exact in zip(torch.autograd.grad(probability, (p, q, zero)), expected_partials, strict=True):
            self.assertAlmostEqual(actual.item(), float(exact))

  def test_anytime_bounds_validates_exclusive_weights_and_group_membership(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    with self.assertRaisesRegex(ValueError, "occurs more than once"):
      ctx.add_facts("choice", [(0.5, 1), (0.5, 2)], disjunctions=[[0, 1], [1]])
    with self.assertRaisesRegex(ValueError, "must have a weight"):
      ctx.add_facts("choice", [(None, 1)], disjunctions=[[0]])
    ctx.add_facts("choice", [(0.75, 1), (0.75, 2)], disjunctions=[[0, 1]])
    with self.assertRaises(ValueError):
      ctx.evaluate_bounds("choice", (1,), max_facts=0)
    value = ctx.evaluate_bounds("choice", (1,), normalize_disjunctions=True).value
    self.assertLessEqual(value.lower, 0.5)
    self.assertGreaterEqual(value.upper, 0.5)
    self.assertAlmostEqual(value.upper - value.lower, 0.0)

  def test_anytime_bounds_exclusive_gradients_propagate_through_logits(self):
    import torch

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("choice", int)
    ctx.add_relation("flag", int)
    ctx.add_rule("hit() = (choice(1) and flag(1)) or (choice(2) and not flag(1))")
    logits = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    p, q = weights = logits.softmax(0).unbind()
    ctx.add_facts("choice", [(p, 1), (q, 2)], disjunctions=[[0, 1]])
    flag = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    ctx.add_facts("flag", [(flag, 1)])
    for budget in (2, 3):
      result = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=budget, normalize_disjunctions=True).value
      self.assertEqual(result.interval.active_facts, budget)
      derivatives = {id(p): -0.25, id(q): 0.25, id(flag): 0.0}
      for gradient in result.gradients:
        expected = derivatives[id(gradient.input)]
        self.assertLessEqual(gradient.lower, expected)
        self.assertGreaterEqual(gradient.upper, expected)
      if budget == 2:
        self.assertIsNone(result.probability)
      else:
        probability = result.probability
        assert probability is not None
        expected = weights[0] * flag + weights[1] * (1 - flag)
        actual_grads = torch.autograd.grad(probability, (logits, flag), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (logits, flag))
        for actual, expected_derivative in zip(actual_grads, expected_grads, strict=True):
          torch.testing.assert_close(actual, expected_derivative)

  def test_anytime_bounds_reject_unsupported_inputs(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    with self.assertRaisesRegex(ValueError, "use evaluate_bounds"):
      ctx.run()

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_program("rel p = {1, 2}\nrel answer(n) = n := count!(x: p(x))")
    with self.assertRaisesRegex(ValueError, "aggregation"):
      ctx.evaluate_bounds("answer", (2,))

    ctx = scallopy.Context.from_mode(scallopy.modes.anytime_bounds)
    ctx.add_relation("witness", int)
    ctx.add_facts("witness", [(float("nan"), 1)])
    with self.assertRaisesRegex(ValueError, "invalid probability"):
      ctx.evaluate_bounds("witness", (1,), max_facts=0)
    with self.assertRaisesRegex(ValueError, "bounds tolerance"):
      ctx.evaluate_bounds("witness", (1,), tolerance=-0.1)

  def test_edge_path(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(1, 2), (2, 3), (3, 4)])
    ctx.add_rule("path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))")
    ctx.run()
    self.assertEqual(list(ctx.relation_tuples("path")), [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

  def test_tropical_edge_path(self):
    ctx = scallopy.ScallopContext(provenance="tropical")
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(1, (1, 2)), (1, (2, 3)), (1, (3, 4)), (1, (1, 4))])
    ctx.add_rule("path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))")
    ctx.run()
    self.assertEqual(list(ctx.relation("path")), [
      scallopy.Row((1, 2), 1), scallopy.Row((1, 3), 2), scallopy.Row((1, 4), 1),
      scallopy.Row((2, 3), 1), scallopy.Row((2, 4), 2),
      scallopy.Row((3, 4), 1),
    ])

  def test_fibonacci(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("fib", (int, int))
    ctx.add_facts("fib", [(0, 1), (1, 1)])
    ctx.add_rule("fib(x, y + z) = fib(x - 1, y), fib(x - 2, z), x <= 5")
    ctx.run()
    self.assertEqual(list(ctx.relation_tuples("fib")), [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)])

  def test_fibonacci_demand(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("fib", (int, int))
    ctx.add_facts("fib", [(0, 1), (1, 1)])
    ctx.add_rule("fib(x, y + z) = fib(x - 1, y), fib(x - 2, z), x > 1", demand="bf")
    ctx.add_rule("result(y) = fib(5, y)")
    ctx.run()
    self.assertEqual(list(ctx.relation_tuples("result")), [(8,)])

  def test_digit_sum(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("digit1", int)
    ctx.add_relation("digit2", int)
    ctx.add_facts("digit1", [(i,) for i in range(10)])
    ctx.add_facts("digit2", [(i,) for i in range(10)])
    ctx.add_rule("sum_2(a + b) = digit1(a), digit2(b)")
    ctx.run()
    self.assertEqual(list(ctx.relation_tuples("sum_2")), [(i,) for i in range(19)])

  def test_run_batch(self):
    ctx = scallopy.Context()
    ctx.add_relation("digit", int)
    ctx.add_rule("successor(x + 1) = digit(x)")
    ctx.compile()
    inputs = {"digit": [[1], [(4,)]]}

    for parallel in (False, True):
      with self.subTest(parallel=parallel):
        batches = ctx.run_batch(inputs, "successor", parallel=parallel)
        self.assertEqual([list(batch) for batch in batches], [[scallopy.Row((2,), None)], [scallopy.Row((5,), None)]])
        batches = ctx.run_batch(inputs, ["digit", "successor"], parallel=parallel)
        self.assertEqual([list(batch.tuples()) for batch in batches], [[(1,)], [(5,)]])

    with self.assertRaisesRegex(ValueError, "Expected 2 output relations"):
      ctx.run_batch(inputs, ["digit"])

  def test_run_batch_with_disjunctions(self):
    ctx = scallopy.Context(provenance="topkproofs")
    ctx.add_relation("choice", int)
    ctx.add_rule("both(x) = choice(x), choice(y), x != y")
    ctx.compile()
    facts = [(0.5, (1,)), (0.5, (2,))]
    inputs = {"choice": [facts, (facts, [[0, 1]])]}

    independent, exclusive = ctx.run_batch(inputs, "both")
    self.assertEqual(list(independent), [scallopy.Row((1,), 0.25), scallopy.Row((2,), 0.25)])
    self.assertEqual(list(exclusive), [])

  def test_softmax(self):
    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("digit", int)
    ctx.add_facts("digit", [(0.5, (i,)) for i in range(10)])
    ctx.add_rule("softmax_digit(n) = n := softmax(d: digit(d))")
    ctx.run()
    for i, row in enumerate(ctx.relation("softmax_digit")):
      self.assertAlmostEqual(row.value, 0.1)
      self.assertEqual(row.tuple, (i,))

  def test_rule_weight(self):
    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("speaks", (int, str))
    ctx.add_relation("lives_in", (int, str))
    ctx.add_facts("speaks", [(None, (0, "english")), (None, (1, "chinese")), (None, (2, "english"))])
    ctx.add_facts("lives_in", [(None, (0, "china")), (None, (1, "us")), (None, (2, "us"))])
    ctx.add_rule("born_in(a, \"china\") = lives_in(a, \"china\")", tag = 0.8)
    ctx.add_rule("born_in(a, \"china\") = speaks(a, \"chinese\")", tag = 0.6)
    ctx.run()
    self.assertEqual(list(ctx.relation("born_in")), [scallopy.Row((0, "china"), 0.8), scallopy.Row((1, "china"), 0.6)])

  def test_proofs_disjunction(self):
    ctx = scallopy.ScallopContext(provenance="proofs")
    ctx.add_relation("assignment", (str, bool))
    ctx.add_relation("bf_var", (int, str))
    ctx.add_relation("bf_not", (int, int))
    ctx.add_relation("bf_and", (int, int, int))
    ctx.add_relation("bf_or", (int, int, int))
    ctx.add_relation("bf_root", (int,)) # Note the type here is a tuple (int,), therefore the facts should contain one-tuples
    ctx.add_rule("eval_bf(bf, res) :- bf_var(bf, v), assignment(v, res)")
    ctx.add_rule("eval_bf(bf, !res) :- bf_not(bf, cbf), eval_bf(cbf, res)")
    ctx.add_rule("eval_bf(bf, r1 && r2) :- bf_and(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("eval_bf(bf, r1 || r2) :- bf_or(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("result(r) :- bf_root(bf), eval_bf(bf, r)")
    ctx.add_facts("bf_var", [(0, "A"), (1, "B")])
    ctx.add_facts("bf_not", [(2, 0), (3, 1)])
    ctx.add_facts("bf_and", [(4, 0, 2), (5, 1, 3)])
    ctx.add_facts("bf_or", [(6, 4, 5)])
    ctx.add_facts("bf_root", [(6,)]) # Note that here (6,) is a one-tuple
    ctx.add_facts("assignment", [("A", False), ("A", True), ("B", False), ("B", True)], disjunctions=[[0, 1], [2, 3]])
    ctx.run()
    self.assertEqual(list(ctx.relation_tuples("result")), [(False,)])

  def test_top_k_proofs_disjunction(self):
    ctx = scallopy.ScallopContext(provenance="topkproofs")
    ctx.add_relation("assignment", (str, bool))
    ctx.add_relation("bf_var", (int, str), non_probabilistic=True)
    ctx.add_relation("bf_not", (int, int), non_probabilistic=True)
    ctx.add_relation("bf_and", (int, int, int), non_probabilistic=True)
    ctx.add_relation("bf_or", (int, int, int), non_probabilistic=True)
    ctx.add_relation("bf_root", int, non_probabilistic=True) # Note that the type here is a single type int, therefore the facts could be either one-tuples or single integer
    ctx.add_rule("eval_bf(bf, res) :- bf_var(bf, v), assignment(v, res)")
    ctx.add_rule("eval_bf(bf, !res) :- bf_not(bf, cbf), eval_bf(cbf, res)")
    ctx.add_rule("eval_bf(bf, r1 && r2) :- bf_and(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("eval_bf(bf, r1 || r2) :- bf_or(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("result(r) :- bf_root(bf), eval_bf(bf, r)")
    ctx.add_facts("bf_var", [(0, "A"), (1, "B")])
    ctx.add_facts("bf_not", [(2, 0), (3, 1)])
    ctx.add_facts("bf_and", [(4, 0, 2), (5, 1, 3)])
    ctx.add_facts("bf_or", [(6, 4, 5)])
    ctx.add_facts("bf_root", [6]) # Note that the fact here is a single integer
    ctx.add_facts("assignment", [(1.0, ("A", False)), (1.0, ("A", True)), (1.0, ("B", False)), (1.0, ("B", True))], disjunctions=[[0, 1], [2, 3]])
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [scallopy.Row((False,), 1.0)])


if __name__ == "__main__":
  unittest.main()
