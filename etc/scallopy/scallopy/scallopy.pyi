from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .attribute import ForeignAttributeProcessor
from .function import ForeignFunction
from .input_output import CSVFileOptions
from .modes import BoundsStop
from .predicate import ForeignPredicate
from .provenance import ScallopProvenance
from .torch_importer import Tensor

_BoundsInterval = Tuple[float, float, int, BoundsStop]
_BoundsResult = Tuple[
  _BoundsInterval,
  Optional[float],
  List[Tuple[Tensor, float, float, Optional[float]]],
  Optional[List[Tuple[int, int, float]]],
]


def torch_tensor_enabled() -> bool:
  """
  Returns a boolean indicating whether this version of `scallopy`
  is compiled with torch tensor enabled.
  """


class InternalBoundsSession:
  def advance(self, max_facts: int, timeout: Optional[float] = None) -> _BoundsInterval: ...


class InternalScallopContext:
  def __init__(
    self,
    provenance: str = "unit",
    k: int = 3,
    wmc_with_disjunctions: bool = False,
    custom_provenance: Optional[ScallopProvenance] = None,
  ) -> None: ...

  def import_file(self, file_name: str) -> None: ...

  def clone(self) -> InternalScallopContext: ...

  def clone_with_new_provenance(
    self,
    provenance: str,
    k: int,
    wmc_with_disjunctions: bool,
  ) -> InternalScallopContext: ...

  def set_non_incremental(self) -> None: ...

  def compile(self) -> None: ...

  def enable_tensor_registry(self) -> None: ...

  def set_debug_front(self, debug_front: bool) -> None: ...

  def set_debug_back(self, debug_back: bool) -> None: ...

  def set_debug_ram(self, debug_ram: bool) -> None: ...

  def set_k(self, k: int) -> None: ...

  def set_early_discard(self, early_discard: bool) -> None: ...

  def set_iter_limit(self, k: int) -> None: ...

  def remove_iter_limit(self) -> None: ...

  def add_monitors(self, monitors: List[str]) -> None: ...

  def load_monitors(self, monitors: List[str]) -> None: ...

  def run(self) -> None: ...

  def run_with_debug_tag(self) -> None: ...

  def evaluate_bounds(
    self,
    relation: str,
    tuple: Tuple[Any, ...],
    tolerance: Optional[float],
    max_facts: int,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight", "required", "lookahead"] = "input",
    checkpoint_schedule: Literal["doubling", "adaptive"] = "doubling",
    *,
    second_order: bool = False,
    gradient_targets: Optional[List[Tuple[Tensor, float]]] = None,
  ) -> _BoundsResult: ...

  def start_bounds(
    self,
    relation: str,
    tuple: Tuple[Any, ...],
    tolerance: float,
    normalize_disjunctions: bool,
    fact_order: Literal["input", "weight", "required", "lookahead"],
    checkpoint_schedule: Literal["doubling", "adaptive"],
  ) -> InternalBoundsSession: ...

  def evaluate_relation_bounds(
    self,
    relation: str,
    tolerance: float,
    max_facts: int,
    normalize_disjunctions: bool,
    fact_order: Literal["input", "weight"],
    checkpoint_schedule: Literal["doubling", "adaptive"],
  ) -> List[Tuple[Tuple[Any, ...], _BoundsInterval]]: ...

  def evaluate_relation_bounds_with_gradients(
    self,
    relation: str,
    max_facts: int,
    normalize_disjunctions: bool,
    fact_order: Literal["input", "weight"],
  ) -> List[Tuple[Tuple[Any, ...], _BoundsResult]]: ...

  def evaluate_bounds_batch(
    self,
    queries: List[Tuple[str, Tuple[Any, ...]]],
    inputs: Dict[str, List[List[Tuple[Any, ...]]]],
    tolerance: Optional[float],
    max_facts: int,
    normalize_disjunctions: bool,
    fact_order: Literal["input", "weight", "required", "lookahead"],
    checkpoint_schedule: Literal["doubling", "adaptive"],
  ) -> List[_BoundsResult]: ...

  def run_batch(
    self,
    output_relations: List[List[str]],
    inputs: Dict[str, List[List[Tuple[Any, ...]]]],
    parallel: bool,
  ) -> List[List[InternalScallopCollection]]: ...

  def add_relation(
    self,
    relation: str,
    load_csv: Optional[Union[CSVFileOptions, str]] = None,
    demand: Optional[str] = None,
  ) -> str: ...

  def add_facts(self, relation: str, elems: List[Tuple]) -> None: ...

  def add_joint_choice(self, outcomes: List[Tuple[Union[float, Tensor], Dict[str, List[Any]]]]) -> None: ...

  def check_tuple(self, relation: str, tup: Tuple) -> bool: ...

  def check_tuples(self, relation: str, tups: List[Tuple]) -> bool: ...

  def add_program(self, program: str) -> None: ...

  def add_item(self, item: str) -> List[str]: ...

  def add_rule(
    self,
    rule: str,
    tag: Optional[Any] = None,
    demand: Optional[str] = None,
  ) -> None: ...

  def register_foreign_function(self, f: ForeignFunction) -> None: ...

  def register_foreign_predicate(self, f: ForeignPredicate) -> None: ...

  def register_foreign_attribute(self, attr: ForeignAttributeProcessor) -> None: ...

  def dump_front_ir(self) -> None: ...

  def get_front_ir(self) -> str: ...

  def input_tags(self) -> Optional[List[Any]]: ...

  def relation(self, r: str) -> InternalScallopCollection: ...

  def relation_with_debug_tag(self, r: str) -> InternalScallopCollection: ...

  def has_relation(self, r: str) -> bool: ...

  def relation_is_computed(self, r: str) -> bool: ...

  def num_relations(self, include_hidden: bool = False) -> int: ...

  def relations(self, include_hidden: bool = False) -> List[str]: ...

  def relation_field_names(self, relation: str) -> Optional[List[Optional[str]]]: ...


class InternalScallopCollection:
  """
  A collection of tuples (maybe associated with tags)
  """
  def num_input_facts(self) -> Optional[int]:
    """
    Get the number of input facts for a valid provenance semiring
    """

  def input_tags(self) -> Optional[List[Any]]:
    """
    Get all the input tags
    """

  def len(self) -> int:
    """
    Get the number of elements in the collection
    """

  def __len__(self) -> int: ...

  def __iter__(self) -> InternalScallopCollectionIterator:
    """
    Iterate through the tuples of the collection
    """

  def iter_tagged(self) -> InternalScallopCollectionIterator: ...


class InternalScallopCollectionIterator:
  """
  An iterator over the scallop collection
  """

  def __iter__(self) -> InternalScallopCollectionIterator: ...

  def __next__(self) -> Tuple[Any, ...]:
    """
    Get the next tuple in the collection
    """
