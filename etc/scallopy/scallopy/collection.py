from dataclasses import dataclass
from typing import Any, Generic, Iterable, Iterator, List, Tuple, TypeVar

from .scallopy import InternalScallopCollection

from . import torch_importer

_ValueT = TypeVar("_ValueT", covariant=True)


@dataclass(frozen=True)
class Row(Generic[_ValueT]):
  """A relation tuple and its recovered provenance value."""

  tuple: Tuple[Any, ...]
  value: _ValueT


class ScallopCollection:
  def __init__(
    self,
    provenance: str,
    internal_collection: InternalScallopCollection,
  ):
    self.provenance = provenance
    self._internal = internal_collection

  def __iter__(self) -> Iterator[Tuple[Any, ...]]:
    if self.provenance in ("unit", "proofs"):
      yield from self._internal
    else:
      yield from self._converted_tags(self._internal)

  def _rows(self) -> Iterator[Row[Any]]:
    if self.provenance == "unit":
      for tup in self._internal:
        yield Row(tup, None)
    else:
      items = self._internal.iter_tagged() if self.provenance == "proofs" else self._internal
      for value, tup in self._converted_tags(items):
        if tup is None:
          raise TypeError("Could not convert a Scallop result tuple to Python")
        yield Row(tup, value)

  def _converted_tags(self, items: Iterable[Tuple[Any, ...]]) -> Iterator[Tuple[Any, Tuple[Any, ...]]]:
    if self.provenance == "diffminmaxprob":
      for ((s, p), t) in items:
        if s == 1:
          yield (p, t)
        elif s == 0:
          yield (torch_importer.torch.tensor(p), t)
        else:
          yield (1 - p, t)
    elif self.provenance == "diffaddmultprob" or \
         self.provenance == "diffnandmultprob" or \
         self.provenance == "diffmaxmultprob" or \
         self.provenance == "diffnandminprob" or \
         self.provenance == "difftopkproofs" or \
         self.provenance == "diffsamplekproofs" or \
         self.provenance == "difftopbottomkclauses":
      input_tags = self._internal.input_tags()
      if input_tags is None:
        raise RuntimeError(f"Missing input tags for {self.provenance}")
      for ((p, deriv), t) in items:
        diff_prob = diff_proofs_prob(p, deriv, input_tags)
        yield (diff_prob, t)
    elif self.provenance == "difftopkproofsdebug":
      input_tags = self._internal.input_tags()
      if input_tags is None:
        raise RuntimeError(f"Missing input tags for {self.provenance}")
      for ((p, deriv, proofs), t) in items:
        diff_prob = diff_proofs_prob(p, deriv, input_tags)
        yield ((diff_prob, proofs), t)
    else:
      for tag, tup in items:
        yield tag, tup


def diff_proofs_prob(p: float, deriv: List[Tuple[int, float]], input_tags: List[torch_importer.Tensor]) -> torch_importer.Tensor:
  def hook(grad):
    for (tag_id, weight) in deriv:
      source_tensor = input_tags[tag_id]
      if source_tensor.requires_grad:
        source_tensor.backward(weight * grad, retain_graph=True)
  v = torch_importer.torch.tensor(p, requires_grad=True)
  v.register_hook(hook)
  return v
