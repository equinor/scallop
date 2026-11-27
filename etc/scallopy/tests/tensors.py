from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import scallopy

class TensorTests(unittest.TestCase):
  @unittest.skipUnless(find_spec("scallop_opencv"), "requires the OpenCV plugin")
  def test_image_download_checks_http_status_before_decoding(self):
    from io import BytesIO

    import requests
    from PIL import Image
    from scallop_opencv.load_image import load_image_url

    url = "https://example.invalid/image.png"
    response = requests.Response()
    response.url = url
    response.status_code = 200
    with BytesIO() as data:
      Image.new("RGBA", (1, 1), (12, 34, 56, 78)).save(data, format="PNG")
      response._content = data.getvalue()
    with patch("requests.get", return_value=response) as get:
      torch.testing.assert_close(load_image_url(url), torch.tensor([[[12, 34, 56]]], dtype=torch.uint8))
      get.assert_called_once_with(url, timeout=30)

    response.status_code = 404
    response._content = b"not an image"
    with patch("requests.get", return_value=response), self.assertRaises(requests.HTTPError):
      load_image_url(url)

    with patch("requests.get", side_effect=requests.Timeout), self.assertRaises(requests.Timeout):
      load_image_url(url)

  @unittest.skipUnless(find_spec("PIL"), "requires Pillow")
  def test_debug_image_saves_jpeg_under_image_id(self):
    from PIL import Image
    from scallopy.utils.image import save_debug_image

    tensor = torch.zeros((2, 3), dtype=torch.uint8)
    with TemporaryDirectory() as root:
      for index in range(2):
        save_debug_image(root, 42, index, "mask", tensor)
        with Image.open(Path(root) / "42" / f"mask-{index}.jpg") as image:
          self.assertEqual(image.format, "JPEG")
          self.assertEqual(image.size, (3, 2))

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_foreign_function_in_parallel_batch(self):
    @scallopy.foreign_function
    def double_tensor(value: scallopy.Tensor) -> scallopy.Tensor:
      return 2 * value

    ctx = scallopy.Context()
    ctx.register_foreign_function(double_tensor)
    ctx.add_relation("batch_input", scallopy.Tensor)
    ctx.add_rule("result($double_tensor(x)) = batch_input(x)")
    ctx._internal.compile()

    values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    inputs = {"batch_input": [[(value,)] for value in values]}
    results = ctx._internal.run_batch([["result"], ["result"]], inputs, True)
    for (collection,), value in zip(results, values, strict=True):
      [(result,)] = list(collection)
      torch.testing.assert_close(result, 2 * value)

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_1(self):
    x, y = torch.randn(5), torch.randn(5)
    s = x.dot(y)

    ctx = scallopy.Context()
    ctx.add_relation("r", (int, scallopy.Tensor))
    ctx.add_facts("r", [(1, x), (2, y)])
    ctx.add_rule("y($dot(ta, tb)) = r(1, ta) and r(2, tb)")

    ctx.run()

    result = next(ctx.relation_tuples("y"))[0]
    assert s == result

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_2(self):
    x, y = torch.randn(5), torch.randn(5)
    gt_sum = x + y

    ctx = scallopy.Context()
    ctx.add_relation("r", (int, scallopy.Tensor))
    ctx.add_facts("r", [(1, x), (2, y)])
    ctx.add_rule("y(ta + tb) = r(1, ta) and r(2, tb)")
    ctx.run()
    my_sum = next(ctx.relation_tuples("y"))[0]

    assert all(gt_sum == my_sum)

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_3(self):
    x = torch.randn(10)
    y = torch.randn(10)
    gt_sim = x.dot(y) / (x.norm() * y.norm()) + 1.0 / 2.0

    ctx = scallopy.Context(provenance="difftopkproofs")
    ctx.add_relation("embed", (int, scallopy.Tensor), non_probabilistic=True)
    ctx.add_facts("embed", [(1, x), (2, y)])
    ctx.add_rule("similar(x, y) = embed(x, tx) and embed(y, ty) and x != y and soft_eq<Tensor>(tx, ty)")
    ctx.run()
    my_sim = next(iter(ctx.relation("similar"))).value

    assert abs(gt_sim.item() - my_sim.item()) < 0.001

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_backprop_4(self):
    torch.manual_seed(1357)
    x = torch.randn(10, requires_grad=True)
    y = torch.randn(10)
    opt = torch.optim.Adam(params=[x], lr=0.01)
    gt_initial_sim = x.dot(y) / (x.norm() * y.norm()) + 1.0 / 2.0

    ctx = scallopy.Context(provenance="difftopkproofs")
    ctx.add_relation("embed", (int, scallopy.Tensor), non_probabilistic=True)
    ctx.add_facts("embed", [(1, x), (2, y)])
    ctx.add_rule("similar(x, y) = embed(x, tx) and embed(y, ty) and x != y and soft_eq<Tensor>(tx, ty)")
    ctx.run()
    my_initial_sim = next(iter(ctx.relation("similar"))).value

    assert abs(gt_initial_sim.item() - my_initial_sim.item()) < 0.001

    # Derive a loss, backward, and step
    l = torch.nn.functional.mse_loss(my_initial_sim, torch.tensor(1.0))
    l.backward()
    opt.step()

    # New similarity
    new_sim = x.dot(y) / (x.norm() * y.norm()) + 1.0 / 2.0
    assert new_sim > my_initial_sim

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_forward_backprop_1(self):
    torch.manual_seed(1357)
    batch_size = 16

    x = torch.randn((batch_size, 10), requires_grad=True)
    y = torch.randn((batch_size, 10), requires_grad=True)
    opt = torch.optim.Adam(params=[x, y], lr=0.01)

    scl_module = scallopy.Module(
      program="""
        type embedding_1(embed: Tensor)
        type embedding_2(embed: Tensor)
        rel similar() = embedding_1(t1) and embedding_2(t2) and soft_eq<Tensor>(t1, t2)
        query similar
      """,
      non_probabilistic=["embedding_1", "embedding_2"],
      output_relation="similar",
      output_mapping=(),
      dispatch="serial")

    def step() -> float:
      result = scl_module(embedding_1=[[(x[i],)] for i in range(batch_size)], embedding_2=[[(y[i],)] for i in range(batch_size)])
      gt = torch.ones(batch_size)
      l = torch.nn.functional.mse_loss(result, gt)
      l.backward()
      opt.step()
      return l.item()

    curr_loss = step()
    for j in range(4):
      next_loss = step()
    assert next_loss <= curr_loss
    curr_loss = next_loss
