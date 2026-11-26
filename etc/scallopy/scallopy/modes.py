"""Typed provenance selections for :meth:`ScallopContext.from_mode`."""

from dataclasses import dataclass
from math import inf, isfinite, nextafter
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from . import torch_importer
from .provenance import ScallopProvenance
from .torch_importer import Tensor

_FactT = TypeVar("_FactT")
_ValueT = TypeVar("_ValueT")

_Fact: TypeAlias = tuple[Any, ...] | int | float | str | Tensor
_ScoreFact: TypeAlias = tuple[float | None, _Fact]
_TensorFact: TypeAlias = tuple[Tensor | None, _Fact]
_DebugFact: TypeAlias = tuple[tuple[Tensor, int] | None, _Fact]
ProofSet: TypeAlias = list[list[int]]
DebugProofSet: TypeAlias = list[list[tuple[bool, int]]]
BoundsStop: TypeAlias = Literal["tolerance", "fact_limit", "time_limit", "complete"]


@dataclass(frozen=True)
class ProbabilityInterval:
  """An outward-rounded enclosure of a ground query's probability."""

  lower: float
  upper: float
  active_facts: int
  stop_reason: BoundsStop


@dataclass(frozen=True)
class GradientInterval:
  """An enclosure of the partial derivative for one input fact's weight."""

  input: Tensor
  lower: float
  upper: float


@dataclass(frozen=True)
class ApproximateGradient:
  """A fact-weight partial used by approximate backward and its absolute error bound."""

  input: Tensor
  estimate: float
  absolute_error: float


@dataclass(frozen=True)
class ApproximateBounds:
  """A differentiable midpoint estimate, with probability and partial error bounds."""

  estimate: Tensor
  absolute_error: float
  gradients: tuple[ApproximateGradient, ...]


def _absolute_error(estimate: float, lower: float, upper: float) -> float:
  radius = max(abs(estimate - lower), abs(upper - estimate))
  return nextafter(radius, inf) if radius else 0.0


@dataclass(frozen=True)
class DifferentiableBounds:
  """Certified probability and partial derivatives; autograd when inference is exact."""

  interval: ProbabilityInterval
  gradients: tuple[GradientInterval, ...]
  probability: Tensor | None

  def approximate_backward(self) -> ApproximateBounds:
    """Use midpoint partials for first-order backward, even if inference is unresolved."""
    approximate = []
    for gradient in self.gradients:
      midpoint = 0.5 * gradient.lower + 0.5 * gradient.upper
      used = float(gradient.input.detach().new_tensor(midpoint).item())
      if not isfinite(used):
        raise ValueError("gradient midpoint is not finite in the fact weight's dtype")
      approximate.append(ApproximateGradient(
        gradient.input, used, _absolute_error(used, gradient.lower, gradient.upper),
      ))
    lower, upper = self.interval.lower, self.interval.upper
    midpoint = 0.5 * lower + 0.5 * upper
    estimate = _autograd_probability(midpoint, [(grad.input, grad.estimate) for grad in approximate])
    return ApproximateBounds(estimate, _absolute_error(midpoint, lower, upper), tuple(approximate))


def _autograd_probability(
  value: float, partials: list[tuple[Tensor, float]], hessian: list[tuple[int, int, float]] | None = None,
) -> Tensor:
  torch = torch_importer.torch
  device = partials[0][0].device if partials else None
  probability = torch.tensor(value, dtype=torch.float64, device=device)
  if not partials:
    return probability
  sources = [source.reshape(()).to(device=device, dtype=torch.float64) for source, _ in partials]
  derivatives = [derivative for _, derivative in partials]

  class QueryProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, result, *inputs):
      if hessian is not None:
        ctx.save_for_backward(*inputs)
      return result

    @staticmethod
    def backward(ctx, *grad_outputs):
      (grad,) = grad_outputs
      if hessian is None:
        return (None, *(grad * derivative for derivative in derivatives))
      return (None, *QueryPartials.apply(grad, *ctx.saved_tensors).unbind())

  class QueryPartials(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, *inputs):
      ctx.save_for_backward(grad)
      return grad * grad.new_tensor(derivatives)

    @staticmethod
    def backward(ctx, *grad_outputs):
      return QueryPartials._double_backward(ctx, *grad_outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def _double_backward(ctx, grad_grad):
      (grad,) = ctx.saved_tensors
      weighted = [torch.zeros_like(grad) for _ in derivatives]
      assert hessian is not None
      for i, j, entry in hessian:
        weighted[i] = weighted[i] + entry * grad_grad[j]
        if i != j:
          weighted[j] = weighted[j] + entry * grad_grad[i]
      return (torch.dot(grad_grad, grad_grad.new_tensor(derivatives)), *(grad * term for term in weighted))

  return QueryProbability.apply(probability, *sources)


@dataclass(frozen=True)
class ProvenanceMode(Generic[_FactT, _ValueT]):
  """The input fact and recovered output types of a provenance."""

  _name: str
  _k: int = 3
  _wmc_with_disjunctions: bool = False
  _custom: ScallopProvenance | None = None


unit = ProvenanceMode[_Fact, None]("unit")
proofs = ProvenanceMode[_Fact, ProofSet]("proofs")
tropical = ProvenanceMode[tuple[int | None, _Fact], int]("tropical")
min_max_prob = ProvenanceMode[_ScoreFact, float]("minmaxprob")
add_mult_prob = ProvenanceMode[_ScoreFact, float]("addmultprob")
anytime_bounds = ProvenanceMode[tuple[float | Tensor | None, _Fact], ProbabilityInterval]("anytimebounds")
diff_min_max_prob = ProvenanceMode[_TensorFact, Tensor]("diffminmaxprob")
diff_add_mult_prob = ProvenanceMode[_TensorFact, Tensor]("diffaddmultprob")
diff_nand_mult_prob = ProvenanceMode[_TensorFact, Tensor]("diffnandmultprob")
diff_max_mult_prob = ProvenanceMode[_TensorFact, Tensor]("diffmaxmultprob")
diff_nand_min_prob = ProvenanceMode[_TensorFact, Tensor]("diffnandminprob")


def top_k_proofs(k: int = 3, *, wmc_with_disjunctions: bool = False) -> ProvenanceMode[_ScoreFact, float]:
  return ProvenanceMode("topkproofs", k, wmc_with_disjunctions)


def top_bottom_k_clauses(k: int = 3, *, wmc_with_disjunctions: bool = False) -> ProvenanceMode[_ScoreFact, float]:
  return ProvenanceMode("topbottomkclauses", k, wmc_with_disjunctions)


def diff_sample_k_proofs(k: int = 3) -> ProvenanceMode[_TensorFact, Tensor]:
  return ProvenanceMode("diffsamplekproofs", k)


def diff_top_k_proofs(k: int = 3, *, wmc_with_disjunctions: bool = False) -> ProvenanceMode[_TensorFact, Tensor]:
  return ProvenanceMode("difftopkproofs", k, wmc_with_disjunctions)


def diff_top_bottom_k_clauses(k: int = 3, *, wmc_with_disjunctions: bool = False) -> ProvenanceMode[_TensorFact, Tensor]:
  return ProvenanceMode("difftopbottomkclauses", k, wmc_with_disjunctions)


def diff_top_k_proofs_debug(
  k: int = 3,
  *,
  wmc_with_disjunctions: bool = False,
) -> ProvenanceMode[_DebugFact, tuple[Tensor, DebugProofSet]]:
  return ProvenanceMode("difftopkproofsdebug", k, wmc_with_disjunctions)


def custom(provenance: ScallopProvenance) -> ProvenanceMode[tuple[Any, _Fact], Any]:
  return ProvenanceMode("custom", _custom=provenance)
