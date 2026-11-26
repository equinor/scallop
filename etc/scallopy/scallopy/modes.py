"""Typed provenance selections for :meth:`ScallopContext.from_mode`."""

from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

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
