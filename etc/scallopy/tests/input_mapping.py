import unittest
import torch

import scallopy

class TestInputMapping(unittest.TestCase):
  def test_construct_list_1(self):
    im = scallopy.InputMapping(list(range(10)))
    assert im.kind == "list"
    assert im.shape == (10,)
    assert im.dimension == 1
    assert im.is_singleton == True

  def test_construct_list_1_1(self):
    im = scallopy.InputMapping(range(10))
    assert im.kind == "list"
    assert im.shape == (10,)
    assert im.dimension == 1
    assert im.is_singleton == True

  def test_construct_list_2(self):
    im = scallopy.InputMapping([(i,) for i in range(10)])
    assert im.kind == "list"
    assert im.shape == (10,)
    assert im.dimension == 1
    assert im.is_singleton == False

  def test_construct_list_3(self):
    im = scallopy.InputMapping([(i, j) for i in range(5) for j in range(5)])
    assert im.kind == "list"
    assert im.shape == (25,)
    assert im.dimension == 1
    assert im.is_singleton == False

  def test_construct_list_4(self):
    im = scallopy.InputMapping([[(i, j) for i in range(5)] for j in range(5)])
    assert im.kind == "list"
    assert im.shape == (5, 5)
    assert im.dimension == 2
    assert im.is_singleton == False

  @unittest.expectedFailure
  def test_construct_list_failure_1(self):
    # Tuple size mismatch
    _ = scallopy.InputMapping([(1, 2), (1,)])

  @unittest.expectedFailure
  def test_construct_list_failure_2(self):
    # Empty mapping
    _ = scallopy.InputMapping([])

  @unittest.expectedFailure
  def test_construct_list_failure_3(self):
    # Empty mapping on other dimensions
    _ = scallopy.InputMapping([[], [], []])

  @unittest.expectedFailure
  def test_construct_list_failure_4(self):
    # Unmatched dimensions
    _ = scallopy.InputMapping([[3, 5, 8], [3, 5, 9], [3, 5]])

  @unittest.expectedFailure
  def test_cannot_set_property(self):
    im = scallopy.InputMapping(range(10))
    im.dimension = (1, 1, 1, 1)

  def test_construct_tuple_1(self):
    im = scallopy.InputMapping(())
    assert im.kind == "tuple"
    assert im.shape == ()
    assert im.dimension == 0
    assert im.is_singleton == False

  def test_construct_tuple_2(self):
    im = scallopy.InputMapping((3, 5))
    assert im.kind == "tuple"
    assert im.shape == ()
    assert im.dimension == 0
    assert im.is_singleton == False

  @unittest.expectedFailure
  def test_construct_tuple_failure_1(self):
    _ = scallopy.InputMapping((3, 5, []))

  def test_construct_value_1(self):
    im = scallopy.InputMapping(3)
    assert im.kind == "value"
    assert im.shape == ()
    assert im.dimension == 0
    assert im.is_singleton == True

  def test_construct_dict_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)})
    assert im.kind == "dict"
    assert im.shape == (5, 5)
    assert im.dimension == 2
    assert im.is_singleton == False

  def test_construct_dict_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5), 2: range(2)})
    assert im.kind == "dict"
    assert im.shape == (5, 5, 2)
    assert im.dimension == 3
    assert im.is_singleton == False

  def test_construct_dict_3(self):
    im = scallopy.InputMapping({0: range(3), 1: ["red", "green", "blue"]})
    assert im.kind == "dict"
    assert im.shape == (3, 3)
    assert im.dimension == 2
    assert im.is_singleton == False

  @unittest.expectedFailure
  def test_construct_dict_failure_1(self):
    _ = scallopy.InputMapping({})

  @unittest.expectedFailure
  def test_construct_dict_failure_2(self):
    _ = scallopy.InputMapping({1: range(3)})

  @unittest.expectedFailure
  def test_construct_dict_failure_3(self):
    _ = scallopy.InputMapping({0: range(3), 1: []})

  @unittest.expectedFailure
  def test_construct_dict_failure_4(self):
    _ = scallopy.InputMapping({0: [(1, 3, 5)]})

  @unittest.expectedFailure
  def test_construct_dict_failure_5(self):
    _ = scallopy.InputMapping({"1": [3]})

  @unittest.expectedFailure
  def test_construct_dict_failure_6(self):
    _ = scallopy.InputMapping({-10: [3]})

  def test_process_tensor_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)})
    r = im.process_tensor(torch.zeros((5, 5)))
    assert len(r) == 25

  def test_process_tensor_2(self):
    im = scallopy.InputMapping([[(i, j) for j in range(5)] for i in range(5)])
    r = im.process_tensor(torch.zeros((5, 5)), batched=True)
    assert len(r) == 1
    assert len(r[0]) == 25

  def test_process_tensor_3(self):
    im = scallopy.InputMapping(range(10))
    r = im.process_tensor(torch.randn((16, 10)))
    assert len(r) == 16
    assert len(r[0]) == 10

  def test_retain_k_1(self):
    im = scallopy.InputMapping(range(10), retain_k=3)
    r = im.process_tensor(torch.randn((10,)))
    assert len(r) == 3

  @unittest.expectedFailure
  def test_retain_k_2(self):
    _ = scallopy.InputMapping(range(10), retain_k=3, sample_dim=1)

  @unittest.expectedFailure
  def test_retain_k_3(self):
    _ = scallopy.InputMapping(range(10), retain_k=3, sample_dim=-10)

  def test_categorical_retain_k_1(self):
    im = scallopy.InputMapping(range(10), retain_k=3, sample_strategy="categorical")
    r = im.process_tensor(torch.softmax(torch.randn((10,)), dim=0))
    assert len(r) <= 3 # less than or equal since this is categorical sampling

  def test_categorical_retain_k_2(self):
    im = scallopy.InputMapping({0: range(10), 1: range(10)}, retain_k=3, sample_strategy="categorical")
    r = im.process_tensor(torch.softmax(torch.randn((10, 10)), dim=1))
    assert len(r) <= 3 # less than or equal since this is categorical sampling

  def test_mult_dim_retain_k_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, retain_k=3)
    r = im.process_tensor(torch.randn((5, 5)))
    assert len(r) == 3

  def test_flat_top_k_keeps_mapping_order_and_disjunction_ids(self):
    mapping = scallopy.InputMapping(
      [["a", "b"], ["c", "d"], ["e", "f"]],
      retain_k=3,
      retain_threshold=0.5,
      disjunctive_dim=1,
      supports_disjunctions=True,
    )
    facts = mapping.process_tensor(torch.tensor([[0.5, 0.1], [0.2, 0.0], [0.3, 0.9]]))
    self.assertEqual([(group, fact) for (_, group), fact in facts], [(0, ("a",)), (2, ("f",))])

  def test_empty_top_k_still_applies_threshold(self):
    mapping = scallopy.InputMapping(["a", "b"], retain_k=0, retain_threshold=0.5)
    self.assertEqual(mapping.process_tensor(torch.tensor([0.75, 0.25])), [(0.75, ("a",))])

  def test_top_k_scalar_mapping(self):
    mapping = scallopy.InputMapping("only", retain_k=1)
    self.assertEqual(mapping.process_tensor(torch.tensor(0.75)), [(0.75, ("only",))])

  def test_threshold_keeps_equal_and_nan_tags_with_gradients(self):
    mapping = scallopy.InputMapping([["a", "b"], ["c", "d"]], retain_threshold=0.5)
    values = torch.tensor([[0.5, 0.8], [float("nan"), 0.4]], requires_grad=True)
    facts = mapping.process_tensor(values.t())
    self.assertEqual([fact for _, fact in facts], [("a",), ("b",), ("c",)])
    self.assertTrue(torch.isnan(facts[1][0]))
    torch.stack([tag for tag, _ in facts]).sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))

  def test_threshold_drops_sampled_scalar_below_cutoff(self):
    mapping = scallopy.InputMapping("only", retain_k=1, retain_threshold=0.5)
    self.assertEqual(mapping.process_tensor(torch.tensor(0.4)), [])
    self.assertEqual(mapping.process_tensor(torch.tensor(0.5)), [(0.5, ("only",))])

  def test_mult_dim_retain_k_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_k=2, sample_dim=1)
    r = im.process_tensor(torch.randn((5, 3)))
    assert len(r) == 10

  def test_mult_dim_retain_k_3(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_k=2, sample_dim=0)
    r = im.process_tensor(torch.randn((5, 3)))
    assert len(r) == 6

  def test_mult_dim_categorical_retain_k_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_k=2, sample_dim=0, sample_strategy="categorical")
    i = torch.nn.functional.softmax(torch.randn((5, 3)), dim=0)
    r = im.process_tensor(i)
    assert len(r) <= 2 * 3 # less than or equal since this is categorical sampling

  def test_mult_dim_categorical_retain_k_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3), 2: range(7)}, retain_k=2, sample_dim=0, sample_strategy="categorical")
    i = torch.nn.functional.softmax(torch.randn((5, 3, 7)), dim=0)
    r = im.process_tensor(i)
    assert len(r) <= 2 * 3 * 7 # less than or equal since this is categorical sampling

  def test_retain_threshold_1(self):
    im = scallopy.InputMapping(range(10), retain_threshold=0.5)
    t = torch.randn(10)
    r = im.process_tensor(t)
    assert len(r) == len(t[t > 0.5])

  def test_retain_threshold_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_threshold=0.5)
    t = torch.randn((5, 3))
    r = im.process_tensor(t)
    assert len(r) == len(t[t > 0.5])

  def test_disjunction_1(self):
    im = scallopy.InputMapping(range(10), disjunctive=True, supports_disjunctions=True)
    t = torch.randn((10,))
    r = im.process_tensor(t)
    for ((_, did), _) in r:
      assert did == 0

  def test_disjunction_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, disjunctive=True, supports_disjunctions=True)
    t = torch.randn((5, 5))
    r = im.process_tensor(t)
    for ((_, did), _) in r:
      assert did == 0

  def test_disjunction_3(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, disjunctive_dim=1, supports_disjunctions=True)
    t = torch.randn((5, 5))
    r = im.process_tensor(t)
    for ((_, did), (i, _)) in r:
      assert did == i

  def test_disjunction_4(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, disjunctive_dim=0, supports_disjunctions=True)
    t = torch.randn((5, 5))
    r = im.process_tensor(t)
    for ((_, did), (_, j)) in r:
      assert did == j
