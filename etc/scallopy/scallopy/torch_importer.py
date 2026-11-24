"""Load PyTorch on demand so ordinary Scallop programs don't require it."""

from functools import cache
import sys
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import torch as torch
  from torch import Tensor as Tensor
  from torch.nn import Module as Module

class _DummyTensor: pass


@cache
def _load_torch() -> ModuleType | None:
  try:
    import torch
  except ModuleNotFoundError as error:
    if error.name != "torch":
      raise
    return None
  return torch


def has_pytorch() -> bool:
  return _load_torch() is not None


def __getattr__(name: str):
  if name == "Tensor":
    # Annotations and schema checks shouldn't trigger a Torch import.
    module = _load_torch() if "torch" in sys.modules else None
    return _DummyTensor if module is None else module.Tensor
  if name in ("torch", "Module"):
    module = _load_torch()
    if module is None:
      raise ModuleNotFoundError("PyTorch is required for this feature; install torch", name="torch")
    return module if name == "torch" else module.nn.Module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
