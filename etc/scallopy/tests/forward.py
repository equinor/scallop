import unittest

import scallopy
import torch

class TestForward(unittest.TestCase):
  def test_removed_jit_options_explain_the_migration(self):
    ctx = scallopy.Context(provenance="diffminmaxprob")
    message = "Native JIT was removed.*RAM interpreter"
    with self.assertRaisesRegex(ValueError, message):
      ctx.forward_function(jit=True)
    with self.assertRaisesRegex(ValueError, message):
      ctx.forward_function(jit_name="saved")
    with self.assertRaisesRegex(ValueError, message):
      ctx.forward_function(recompile=True)
    with self.assertRaisesRegex(ValueError, message):
      scallopy.Module(jit=True)
    with self.assertRaisesRegex(ValueError, message):
      scallopy.Module(jit_recompile=True)

  def test_digit_sum_values_and_gradients_across_dispatch_modes(self):
    ctx = scallopy.Context(provenance="diffminmaxprob")
    ctx.add_relation("left", int, range(2))
    ctx.add_relation("right", int, range(2))
    ctx.add_rule("sum_2(a + b) = left(a) and right(b)")

    for dispatch in ("single", "serial", "parallel"):
      with self.subTest(dispatch=dispatch):
        forward = ctx.forward_function("sum_2", range(3), dispatch=dispatch)
        left = torch.tensor([[0.8, 0.3]], requires_grad=True)
        right = torch.tensor([[0.6, 0.2]], requires_grad=True)
        result = forward(left=left, right=right)
        torch.testing.assert_close(result, torch.tensor([[0.6, 0.3, 0.2]]))
        left_grad, right_grad = torch.autograd.grad(result.sum(), (left, right))
        torch.testing.assert_close(left_grad, torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(right_grad, torch.tensor([[1.0, 1.0]]))

  def test_rejects_mixed_tensor_and_list_tag_dtypes(self):
    for provenance in ("diffminmaxprob", "difftopkproofs"):
      with self.subTest(provenance=provenance):
        ctx = scallopy.Context(provenance=provenance)
        ctx.add_relation("left", int, range(1))
        ctx.add_relation("right", int)
        ctx.add_rule("answer(x) = left(x) or right(x)")
        forward = ctx.forward_function("answer", [0], dispatch="serial")

        with self.assertRaisesRegex(TypeError, r"torch\.float32, got torch\.float64 in relation 'right' at batch index 0"):
          forward(
            left=torch.tensor([[0.5]], dtype=torch.float32),
            right=[[(torch.tensor(0.25, dtype=torch.float64), (0,))]],
          )

  def test_rejects_mixed_forward_and_context_proof_tag_dtypes(self):
    ctx = scallopy.Context(provenance="difftopkproofs")
    ctx.add_relation("value", int)
    ctx.add_facts("value", [(torch.tensor(0.25, dtype=torch.float64), (1,))])
    ctx.add_rule("answer(x) = value(x)")
    forward = ctx.forward_function("answer", [1, 2], dispatch="serial")

    with self.assertRaisesRegex(TypeError, r"Expected proof tags with dtype torch\.float64, got torch\.float32"):
      forward(value=[[(torch.tensor(0.5, dtype=torch.float32), (2,))]])

  def test_ragged_proof_tags_keep_gradients_with_an_empty_first_row(self):
    ctx = scallopy.Context(provenance="difftopkproofs")
    ctx.add_relation("value", int)
    ctx.add_rule("answer(x) = value(x)")
    for sparse in (False, True):
      with self.subTest(sparse=sparse):
        forward = ctx.forward_function("answer", [1, 2], dispatch="serial", sparse_jacobian=sparse)
        a, b, c = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True).unbind()

        answer = forward(value=[[], [(a, (1,)), (b, (2,))], [(c, (2,))]])

        torch.testing.assert_close(answer, answer.new_tensor([[0, 0], [0.25, 0.5], [0, 0.75]]))
        for tag, grad in zip((a, b, c), torch.autograd.grad(answer.sum(), (a, b, c)), strict=True):
          torch.testing.assert_close(grad, torch.ones_like(tag))

  def test_top_k_sample(self):
    # First setup the symbols
    symbols = [str(s) for s in range(10)] + ["+", "-", "*", "/"]

    # Next setup the forward function
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.add_relation("symbol", (int, str), [(i, s) for i in range(3) for s in symbols])
    ctx.add_rule("sampled_symbol(id, sym) :- sym = top<3>(s: symbol(id, s))")
    forward = ctx.forward_function("sampled_symbol")

    # Setup the input vectors
    predictions = torch.softmax(torch.randn((16, 3, 14)), dim=2)
    symbol = predictions.reshape((16, -1))
    (_, sampled_symbol_prob) = forward(symbol=symbol)

    # There should be 9 non-zeros in each row: 3 digits * top-3 per digit
    non_zeros = torch.count_nonzero(sampled_symbol_prob, dim=1)
    for x in non_zeros:
      assert int(x) == 9


class TestDigitForward(unittest.TestCase):
  def setUp(self):
    self.ctx = scallopy.ScallopContext(provenance="diffminmaxprob")
    self.ctx.add_relation("digit_1", int, range(10))
    self.ctx.add_relation("digit_2", int, range(10))
    self.ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
    self.ctx.add_rule("mult_2(a * b) = digit_1(a) and digit_2(b)")

  @unittest.expectedFailure
  def test_unknown_relation_1(self):
    self.ctx.forward_function("add_3", list(range(28)))

  def test_normal(self):
    forward = self.ctx.forward_function("sum_2", list(range(19)))
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    sum_2 = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(sum_2.shape, (16, 19))

  def test_no_output_mapping(self):
    forward = self.ctx.forward_function("sum_2")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    (result_mapping, result_tensor) = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(set(result_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_single_dispatch(self):
    forward = self.ctx.forward_function("sum_2", dispatch="single")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    (result_mapping, result_tensor) = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(set(result_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_serial_dispatch(self):
    forward = self.ctx.forward_function("sum_2", dispatch="serial")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    (result_mapping, result_tensor) = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(set(result_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_multi_result(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))})
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    sum_2 = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(sum_2.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))

  @unittest.expectedFailure
  def test_forward_without_output_mapping(self):
    forward = self.ctx.forward_function()
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    _ = forward(digit_1=digit_1, digit_2=digit_2)

  def test_forward_with_output_mapping(self):
    forward = self.ctx.forward_function("sum_2")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result_tensor = forward(
      digit_1=digit_1, digit_2=digit_2,
      output_mappings=[(i,) for i in range(19)])
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_forward_with_output_mapping_2(self):
    forward = self.ctx.forward_function()
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    results = forward(
      digit_1=digit_1,
      digit_2=digit_2,
      output_relations=[["sum_2", "mult_2"]] * 16,
      output_mappings={
        "sum_2": [(i,) for i in range(19)],
        "mult_2": [(i,) for i in range(82)],
      })
    self.assertEqual(results["sum_2"].shape, (16, 19))
    self.assertEqual(results["mult_2"].shape, (16, 82))

  def test_multi_result_single_dispatch(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, dispatch="single")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    sum_2 = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(sum_2.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))

  def test_multi_result_non_parallel_dispatch(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, dispatch="serial")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    sum_2 = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(sum_2.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))

  def test_multi_result_maybe_with_output_mapping(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": None, "mult_2": list(range(100))})
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    (sum_2_mapping, sum_2_tensor) = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(set(sum_2_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(sum_2_tensor.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))



class TestDirectForward(unittest.TestCase):
  def test_forward_hooks_on_both_modules_preserve_gradients(self):
    compute = scallopy.Module(
      program="type value(usize)\nrel answer(x) = value(x)",
      input_mappings={"value": [0]},
      output_mappings={"answer": [0]},
    )
    weights = torch.tensor([[0.75]], requires_grad=True)
    calls = []

    def record(module, args, kwargs, output):
      calls.append(module)
      self.assertIs(kwargs["value"], weights)

    with compute.register_forward_hook(record, with_kwargs=True), \
         compute.forward_fn.register_forward_hook(record, with_kwargs=True):
      result = compute(value=weights)
    self.assertEqual(calls, [compute.forward_fn, compute])
    torch.testing.assert_close(result, weights)
    result.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))

  def test_forward(self):
    sum_2_program = """
    type digit_a(usize), digit_b(usize)
    rel sum_2(a + b) = digit_a(a), digit_b(b)
    """
    compute_sum_2 = scallopy.ScallopForwardFunction(
      program=sum_2_program,
      provenance="difftopkproofs",
      input_mappings={"digit_a": list(range(10)), "digit_b": list(range(10))},
      output_mappings={"sum_2": list(range(19))},
    )
    digit_a, digit_b = torch.randn((16, 10)), torch.randn((16, 10))
    result = compute_sum_2(digit_a=digit_a, digit_b=digit_b)
    self.assertEqual(result.shape, (16, 19))

  def test_forward_with_probabilities(self):
    def process_input(digit_tensor):
      r = []
      (batch_size, _) = digit_tensor.shape
      for task_id in range(batch_size):
        r.append([(p, (i,)) for (i, p) in enumerate(digit_tensor[task_id])])
      return r

    sum_2_program = """
    type digit_a(usize), digit_b(usize)
    rel sum_2(a + b) = digit_a(a), digit_b(b)
    """
    compute_sum_2 = scallopy.ScallopForwardFunction(
      program=sum_2_program,
      provenance="difftopkproofs",
      input_mappings={"digit_a": None, "digit_b": None},
      output_mappings={"sum_2": list(range(19))},
    )
    digit_a, digit_b = torch.randn((16, 10)), torch.randn((16, 10))
    result = compute_sum_2(digit_a=process_input(digit_a), digit_b=process_input(digit_b))
    self.assertEqual(result.shape, (16, 19))

  def test_forward_with_non_probabilistic(self):
    edge_path_program = """
    type edge(usize, usize)
    rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
    """
    compute_path = scallopy.ScallopForwardFunction(
      program=edge_path_program,
      provenance="difftopkproofs",
      non_probabilistic=["edge"],
      output_relation="path")
    edges = [[(0, 1), (1, 2)]]
    output_mapping, result = compute_path(edge=edges)
    self.assertEqual(set(output_mapping), set([(0, 1), (0, 2), (1, 2)]))
    self.assertEqual(result.shape, (1, 3))

if __name__ == "__main__":
  unittest.main()
