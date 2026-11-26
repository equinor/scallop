import subprocess
import sys
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
            assert list(ctx.relation("numbers")) == [(1,), (2,)]
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
    self.assertEqual(list(ctx.relation("row")), [fact])

  def test_typed_probability_rows_and_legacy_results(self):
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
    self.assertEqual(list(ctx.relation("result")), [(0.5, (1,))])
    (cloned_row,) = ctx.clone().relation_rows("result")
    assert_type(cloned_row.value, float)
    self.assertEqual(cloned_row, row)

  def test_typed_proof_rows_preserve_legacy_untagged_iteration(self):
    ctx = scallopy.Context.from_mode(scallopy.modes.proofs)
    ctx.add_relation("choice", int)
    ctx.add_facts("choice", [1, 2], disjunctions=[[0, 1]])
    ctx.run()

    first, second = ctx.relation_rows("choice")
    assert_type(first.value, list[list[int]])
    self.assertEqual((first.tuple, second.tuple), ((1,), (2,)))
    ((first_id,),), ((second_id,),) = first.value, second.value
    self.assertNotEqual(first_id, second_id)
    self.assertEqual(list(ctx.relation("choice")), [(1,), (2,)])

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
    self.assertEqual(list(ctx.relation("input")), [({"value": "tag"}, (1,))])

  def test_edge_path(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(1, 2), (2, 3), (3, 4)])
    ctx.add_rule("path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))")
    ctx.run()
    self.assertEqual(list(ctx.relation("path")), [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

  def test_tropical_edge_path(self):
    ctx = scallopy.ScallopContext(provenance="tropical")
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(1, (1, 2)), (1, (2, 3)), (1, (3, 4)), (1, (1, 4))])
    ctx.add_rule("path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))")
    ctx.run()
    self.assertEqual(list(ctx.relation("path")), [
      (1, (1, 2)), (2, (1, 3)), (1, (1, 4)),
      (1, (2, 3)), (2, (2, 4)),
      (1, (3, 4)),
    ])

  def test_fibonacci(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("fib", (int, int))
    ctx.add_facts("fib", [(0, 1), (1, 1)])
    ctx.add_rule("fib(x, y + z) = fib(x - 1, y), fib(x - 2, z), x <= 5")
    ctx.run()
    self.assertEqual(list(ctx.relation("fib")), [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)])

  def test_fibonacci_demand(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("fib", (int, int))
    ctx.add_facts("fib", [(0, 1), (1, 1)])
    ctx.add_rule("fib(x, y + z) = fib(x - 1, y), fib(x - 2, z), x > 1", demand="bf")
    ctx.add_rule("result(y) = fib(5, y)")
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [(8,)])

  def test_digit_sum(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("digit1", int)
    ctx.add_relation("digit2", int)
    ctx.add_facts("digit1", [(i,) for i in range(10)])
    ctx.add_facts("digit2", [(i,) for i in range(10)])
    ctx.add_rule("sum_2(a + b) = digit1(a), digit2(b)")
    ctx.run()
    self.assertEqual(list(ctx.relation("sum_2")), [(i,) for i in range(19)])

  def test_run_batch(self):
    ctx = scallopy.Context()
    ctx.add_relation("digit", int)
    ctx.add_rule("successor(x + 1) = digit(x)")
    ctx.compile()
    inputs = {"digit": [[1], [(4,)]]}

    for parallel in (False, True):
      with self.subTest(parallel=parallel):
        batches = ctx.run_batch(inputs, "successor", parallel=parallel)
        self.assertEqual([list(batch) for batch in batches], [[(2,)], [(5,)]])
        batches = ctx.run_batch(inputs, ["digit", "successor"], parallel=parallel)
        self.assertEqual([list(batch) for batch in batches], [[(1,)], [(5,)]])

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
    self.assertEqual(list(independent), [(0.25, (1,)), (0.25, (2,))])
    self.assertEqual(list(exclusive), [])

  def test_softmax(self):
    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("digit", int)
    ctx.add_facts("digit", [(0.5, (i,)) for i in range(10)])
    ctx.add_rule("softmax_digit(n) = n := softmax(d: digit(d))")
    ctx.run()
    for (i, (prob, (p_i,))) in enumerate(list(ctx.relation("softmax_digit"))):
      self.assertAlmostEqual(prob, 0.1)
      self.assertEqual(i, p_i)

  def test_rule_weight(self):
    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("speaks", (int, str))
    ctx.add_relation("lives_in", (int, str))
    ctx.add_facts("speaks", [(None, (0, "english")), (None, (1, "chinese")), (None, (2, "english"))])
    ctx.add_facts("lives_in", [(None, (0, "china")), (None, (1, "us")), (None, (2, "us"))])
    ctx.add_rule("born_in(a, \"china\") = lives_in(a, \"china\")", tag = 0.8)
    ctx.add_rule("born_in(a, \"china\") = speaks(a, \"chinese\")", tag = 0.6)
    ctx.run()
    self.assertEqual(list(ctx.relation("born_in")), [(0.8, (0, "china")), (0.6, (1, "china"))])

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
    self.assertEqual(list(ctx.relation("result")), [(False,)])

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
    self.assertEqual(list(ctx.relation("result")), [(1.0, (False,))])


if __name__ == "__main__":
  unittest.main()
