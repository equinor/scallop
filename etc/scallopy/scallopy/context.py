from __future__ import annotations
from sys import maxsize
from typing import Any, Dict, Generic, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, TypeVar, Union, TYPE_CHECKING, cast, overload
from copy import deepcopy

# Try import torch; if not there, delegate to something else
from . import torch_importer

# Import internals
from .scallopy import InternalScallopContext
from .collection import Row, ScallopCollection
from .modes import DifferentiableBounds, GradientInterval, ProbabilityInterval, ProvenanceMode, _Fact, _autograd_probability
from .provenance import ScallopProvenance, DiffAddMultProb2Semiring, DiffNandMultProb2Semiring, DiffMaxMultProb2Semiring
from .input_output import CSVFileOptions
from .input_mapping import InputMapping
from .function import ForeignFunction
from .predicate import ForeignPredicate
from .attribute import ForeignAttributeProcessor
from .utils import Counter, HistoryAction, record_history
from .stdlib import STDLIB

if TYPE_CHECKING:
  from .forward import InternalScallopForwardFunction, _OutputMapping
  from .scallopy import InternalBoundsSession, _BoundsResult
  from .torch_importer import Tensor

_FactT = TypeVar("_FactT")
_ValueT = TypeVar("_ValueT")
_ModeFactT = TypeVar("_ModeFactT")
_ModeValueT = TypeVar("_ModeValueT")
_GroundQuery = Tuple[str, Tuple[Any, ...]]
_BatchInputs = Mapping[str, Sequence[Union[Tuple[List[Any], Optional[List[List[int]]]], List[Any]]]]

class BoundsSession:
  """A resumable ground probability query on a fixed input snapshot."""

  def __init__(self, session: InternalBoundsSession, query: Tuple[Any, ...]):
    self._session = session
    self._query = query

  def advance(self, *, max_facts: int, timeout: Optional[float] = None) -> Row[ProbabilityInterval]:
    """Refine up to a total fact budget and optional timeout in seconds."""
    return Row(self._query, ProbabilityInterval(*self._session.advance(max_facts, timeout)))


# Main context
class ScallopContext(Generic[_FactT, _ValueT]):
  """
  A Scallop execution context that fosters all compilation and execution.

  Usage:

  ``` python
  ctx = ScallopContext()
  ```

  :param provenance: The type of provenance used during execution.
  Default to "unit", and can be any value from the following
  - `"unit"`, no provenance information associated
  - `"proofs"`, collect proofs
  - `"tropical"`, positive integer with +inf; min-add semiring
  - `"minmaxprob"`, min-max probability
  - `"addmultprob"`, add-mult probability
  - `"topkproofs"`, top-k proofs. It is possible to supply a `k` value for this
    provenance
  - `"diffminmaxprob"`, differentiable min-max probability
  - `"diffandmultprob"`, differentiable add-mult probability
  - `"diffandmultprob2"`, differentiable add-mult probability, implemented in Python
  - `"diffnandmultprob2"`, differentiable nand-mult probability, implemented in Python
  - `"diffmaxmultprob2"`, differentiable max-mult probability, implemented in Python
  - `"diffsamplekproofs"`, differentiable unbiased sampling of k proofs. It is
    possible to supply a `k` value for this provenance
  - `"difftopkproofs"`, differentiable top-k proofs. It is possible to supply a
    `k` value for this provenance
  - `"difftopbottomkclauses"`, differentiable top/bottom-k clauses.
    This provenance supports full negation and aggregation. It is possible to supply
    a `k` value for this provenance

  :param k:
  :param train_k:
  :param test_k:
  """
  def __init__(
    self,
    provenance: str = "unit",
    custom_provenance: Optional[ScallopProvenance] = None,
    k: int = 3,
    wmc_with_disjunctions: bool = False,
    train_k: Optional[int] = None,
    test_k: Optional[int] = None,
    fork_from: Optional[ScallopContext[_FactT, _ValueT]] = None,
    no_stdlib: bool = False,
    monitors: List[str] = [],
  ):
    super(ScallopContext, self).__init__()

    # Check if we are creating a forked context or not
    if fork_from is None:
      # Validify provenance; differentiable needs PyTorch
      if "diff" in provenance and not torch_importer.has_pytorch():
        raise Exception("Attempting to use differentiable provenance but with no PyTorch imported")

      # Prepare Python based provenance
      if provenance == "diffaddmultprob2":
        provenance = "custom"
        custom_provenance = DiffAddMultProb2Semiring()
      elif provenance == "diffnandmultprob2":
        provenance = "custom"
        custom_provenance = DiffNandMultProb2Semiring()
      elif provenance == "diffmaxmultprob2":
        provenance = "custom"
        custom_provenance = DiffMaxMultProb2Semiring()
      elif custom_provenance is not None:
        provenance = "custom"
      else:
        pass

      # Setup
      self.provenance = provenance
      self._custom_provenance = custom_provenance
      self._input_mappings = {}
      self._input_retain_topk = {}
      self._input_non_probabilistic = {}
      self._input_is_singleton = {}
      self._mutual_exclusion_counter = Counter()
      self._sample_facts = {}
      self._k = k
      self._wmc_with_disjunctions = wmc_with_disjunctions
      self._train_k = train_k
      self._test_k = test_k
      self._history_actions: List[HistoryAction] = []
      self._monitors = monitors
      self._internal = InternalScallopContext(
        provenance=provenance,
        custom_provenance=custom_provenance,
        k=k,
        wmc_with_disjunctions=wmc_with_disjunctions,
      )

      # Load stdlib
      self._internal.enable_tensor_registry() # Always enable tensor registry for now
      if not no_stdlib:
        self.load_stdlib()

      # Load monitors
      if len(self._monitors) > 0:
        self._internal.add_monitors(self._monitors)
    else:
      # Fork from an existing context
      self.provenance = deepcopy(fork_from.provenance)
      self._custom_provenance = deepcopy(fork_from._custom_provenance)
      self._input_mappings = deepcopy(fork_from._input_mappings)
      self._input_retain_topk = deepcopy(fork_from._input_retain_topk)
      self._input_non_probabilistic = deepcopy(fork_from._input_non_probabilistic)
      self._input_is_singleton = deepcopy(fork_from._input_is_singleton)
      self._mutual_exclusion_counter = deepcopy(fork_from._mutual_exclusion_counter)
      self._sample_facts = deepcopy(fork_from._sample_facts)
      self._k = deepcopy(fork_from._k)
      self._wmc_with_disjunctions = deepcopy(fork_from._wmc_with_disjunctions)
      self._train_k = deepcopy(fork_from._train_k)
      self._test_k = deepcopy(fork_from._test_k)
      self._history_actions = deepcopy(fork_from._history_actions)
      self._monitors = deepcopy(fork_from._monitors)
      self._internal = fork_from._internal.clone()

  @staticmethod
  def from_mode(
    mode: ProvenanceMode[_ModeFactT, _ModeValueT],
    *,
    train_k: Optional[int] = None,
    test_k: Optional[int] = None,
    no_stdlib: bool = False,
    monitors: Sequence[str] = (),
  ) -> ScallopContext[_ModeFactT, _ModeValueT]:
    """Create a context whose fact and result types are inferred from the mode."""
    return ScallopContext(
      provenance=mode._name,
      custom_provenance=mode._custom,
      k=mode._k,
      wmc_with_disjunctions=mode._wmc_with_disjunctions,
      train_k=train_k,
      test_k=test_k,
      no_stdlib=no_stdlib,
      monitors=list(monitors),
    )

  def __getstate__(self):
    """
    Serialize into pickle state
    """
    state = self.__dict__.copy()
    del state["_internal"]
    return state

  def __setstate__(self, state):
    """
    Deserialize from pickle state
    """
    # Serializable part
    self.__dict__.update(state)

    # Initialize empty history actions; this will be populated by later restoration
    self._history_actions: List[HistoryAction] = []

    # Internal scallop context
    self._internal = InternalScallopContext(provenance=self.provenance, custom_provenance=self._custom_provenance, k=self._k, wmc_with_disjunctions=self._wmc_with_disjunctions)

    # Restore from history actions
    for history_action in state["_history_actions"]:
      function = getattr(self, history_action.func_name)
      function(*history_action.pos_args, **history_action.kw_args)

  @overload
  def clone(
    self,
    provenance: None = None,
    k: Optional[int] = None,
    wmc_with_disjunctions: Optional[bool] = None,
    monitors: Optional[List[str]] = None,
  ) -> ScallopContext[_FactT, _ValueT]: ...

  @overload
  def clone(
    self,
    provenance: Optional[str],
    k: Optional[int] = None,
    wmc_with_disjunctions: Optional[bool] = None,
    monitors: Optional[List[str]] = None,
  ) -> ScallopContext[Any, Any]: ...

  def clone(
    self,
    provenance: Optional[str] = None,
    k: Optional[int] = None,
    wmc_with_disjunctions: Optional[bool] = None,
    monitors: Optional[List[str]] = None,
  ) -> ScallopContext[Any, Any]:
    """
    Clone the current context. This is useful for incremental execution:

    ``` python
    ctx2 = ctx.clone()
    ctx2.add_rule("...")
    ctx2.run()

    ctx3 = ctx.clone()
    ctx3.add_rule("...")
    ctx3.run()
    ```

    In this example, `ctx2` and `ctx3` will be executed independently,
    but both could inherit the computation already done on `ctx`.
    """

    # Create a new context
    new_ctx = ScallopContext(fork_from=self)

    if provenance is not None:

      # Clone internal context; this process may fail if the provenance is not compatible
      new_k = k if k is not None else self._k
      new_wmc_with_disjunctions = wmc_with_disjunctions if wmc_with_disjunctions is not None else self._wmc_with_disjunctions
      new_ctx._internal = new_ctx._internal.clone_with_new_provenance(provenance, new_k, new_wmc_with_disjunctions)

      # Update parameters related to provenance
      new_ctx.provenance = provenance
      new_ctx._k = new_k
      new_ctx._wmc_with_disjunctions = new_wmc_with_disjunctions

    if monitors is not None:
      # Update parameters related to provenance
      new_ctx._monitors = monitors
      new_ctx._internal.load_monitors(monitors)

    # Return
    return new_ctx

  def set_debug_front(self, debug_front: bool = True):
    """
    Debug Front
    """
    self._internal.set_debug_front(debug_front)

  def set_debug_back(self, debug_back: bool = True):
    """
    Debug Back
    """
    self._internal.set_debug_back(debug_back)

  def set_debug_ram(self, debug_ram: bool = True):
    """
    Debug RAM
    """
    self._internal.set_debug_ram(debug_ram)

  def set_early_discard(self, early_discard: bool = True):
    """
    Configure the current context to perform early discard (or not)
    """
    self._internal.set_early_discard(early_discard)

  def set_iter_limit(self, iter_limit: Optional[int] = None):
    """
    Configure the current context to have limit on iteration (or not)
    """
    if iter_limit is None:
      self._internal.remove_iter_limit()
    else:
      self._internal.set_iter_limit(iter_limit)

  def run(self) -> None:
    """
    Execute the code under the current context. This operation is incremental
    and might use as many as previous information as possible.

    ``` python
    ctx.run()
    ```
    """
    self._internal.run()

  def evaluate_bounds(
    self,
    relation: str,
    tuple: Tuple[Any, ...],
    *,
    tolerance: float = 0.01,
    max_facts: int = 32,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight", "required", "lookahead"] = "input",
    checkpoint_schedule: Literal["doubling", "adaptive"] = "doubling",
  ) -> Row[ProbabilityInterval]:
    """Evaluate one ground query in an anytime bounds context.

    Set normalize_disjunctions to make input groups exhaustive by normalizing
    each group's weights. Otherwise leftover mass means no outcome occurs.
    In positive queries over independent facts, fact_order can prioritize
    higher weights, facts required by every proof, or a bounded lookahead.
    The adaptive checkpoint schedule can advance faster when bounds stall.
    """
    interval, _, _, _ = self._internal.evaluate_bounds(
      relation, tuple, tolerance, max_facts, normalize_disjunctions, fact_order, checkpoint_schedule,
    )
    return Row(tuple, ProbabilityInterval(*interval))

  def start_bounds(
    self,
    relation: str,
    query: Tuple[Any, ...],
    *,
    tolerance: float = 0.01,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight", "required", "lookahead"] = "input",
    checkpoint_schedule: Literal["doubling", "adaptive"] = "doubling",
  ) -> BoundsSession:
    """Snapshot the query, its program, and input weights for later refinement."""
    session = self._internal.start_bounds(
      relation, query, tolerance, normalize_disjunctions, fact_order, checkpoint_schedule,
    )
    return BoundsSession(session, query)

  def evaluate_bounds_with_gradients(
    self,
    relation: str,
    query: Tuple[Any, ...],
    *,
    max_facts: int = maxsize,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight", "required", "lookahead"] = "input",
    second_order: bool = False,
    gradient_tolerances: Optional[Sequence[Tuple[Tensor, float]]] = None,
    probability_tolerance: Optional[float] = None,
  ) -> Row[DifferentiableBounds]:
    """Bound a ground query and its partials for fact weights that require gradients.

    With normalize_disjunctions, the partials are with respect to the supplied
    weights before group normalization. Set second_order to enable double
    backward for a resolved query. Supply (weight tensor, width) pairs in
    gradient_tolerances to stop when each selected partial meets its target,
    and optionally require a probability interval width as well.
    """
    targets = None
    if gradient_tolerances is not None or probability_tolerance is not None:
      targets = [] if gradient_tolerances is None else list(gradient_tolerances)
    result = self._internal.evaluate_bounds(
      relation, query, probability_tolerance, max_facts, normalize_disjunctions, fact_order,
      second_order=second_order, gradient_targets=targets,
    )
    return Row(query, self._differentiable_bounds(result))

  def evaluate_relation_bounds(
    self,
    relation: str,
    *,
    tolerance: float = 0.01,
    max_facts: int = 32,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight"] = "input",
    checkpoint_schedule: Literal["doubling", "adaptive"] = "doubling",
  ) -> List[Row[ProbabilityInterval]]:
    """Bound every tuple in the relation using a shared fact budget.

    Tuples with a zero upper probability are omitted. An unresolved row may
    still have zero true probability. Tolerance applies to every returned row.
    """
    rows = self._internal.evaluate_relation_bounds(
      relation, tolerance, max_facts, normalize_disjunctions, fact_order, checkpoint_schedule,
    )
    return [Row(query, ProbabilityInterval(*interval)) for query, interval in rows]

  def evaluate_relation_bounds_with_gradients(
    self,
    relation: str,
    *,
    max_facts: int = maxsize,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight"] = "input",
  ) -> List[Row[DifferentiableBounds]]:
    """Bound every row and its fact-weight partials at one shared fact budget.

    Rows may be retained at zero probability when a derivative can be nonzero.
    """
    rows = self._internal.evaluate_relation_bounds_with_gradients(
      relation, max_facts, normalize_disjunctions, fact_order,
    )
    return [Row(query, self._differentiable_bounds(result)) for query, result in rows]

  def evaluate_bounds_batch(
    self,
    queries: Sequence[_GroundQuery],
    *,
    inputs: Optional[_BatchInputs] = None,
    tolerance: float = 0.01,
    max_facts: int = 32,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight", "required", "lookahead"] = "input",
    checkpoint_schedule: Literal["doubling", "adaptive"] = "doubling",
  ) -> List[Row[ProbabilityInterval]]:
    """Evaluate ground queries in order, with independent per-query budgets.

    Each input relation can supply one fact list (or facts and disjunctions)
    per query. Those facts are local to the example and supplement context facts.
    """
    queries = list(queries)
    results = self._internal.evaluate_bounds_batch(
      queries, self._prepare_bounds_batch_inputs(queries, inputs), tolerance, max_facts,
      normalize_disjunctions, fact_order, checkpoint_schedule,
    )
    return [
      Row(query, ProbabilityInterval(*interval))
      for (_, query), (interval, _, _, _) in zip(queries, results, strict=True)
    ]

  def evaluate_bounds_batch_with_gradients(
    self,
    queries: Sequence[_GroundQuery],
    *,
    inputs: Optional[_BatchInputs] = None,
    max_facts: int = maxsize,
    normalize_disjunctions: bool = False,
    fact_order: Literal["input", "weight", "required", "lookahead"] = "input",
  ) -> List[Row[DifferentiableBounds]]:
    """Return certified bounds and gradients for each ground query in order."""
    queries = list(queries)
    results = self._internal.evaluate_bounds_batch(
      queries, self._prepare_bounds_batch_inputs(queries, inputs), None, max_facts,
      normalize_disjunctions, fact_order, "doubling",
    )
    return [
      Row(query, self._differentiable_bounds(result))
      for (_, query), result in zip(queries, results, strict=True)
    ]

  def _prepare_bounds_batch_inputs(
    self, queries: Sequence[_GroundQuery], inputs: Optional[_BatchInputs],
  ) -> Dict[str, List[List[Tuple[Any, ...]]]]:
    if inputs is None:
      return {}
    for relation, examples in inputs.items():
      if len(examples) != len(queries):
        raise ValueError(f"Expected {len(queries)} input examples for {relation!r}, got {len(examples)}")
    return {
      relation: [
        self._prepare_facts(relation, *example) if isinstance(example, tuple) else self._prepare_facts(relation, example)
        for example in examples
      ]
      for relation, examples in inputs.items()
    }

  @staticmethod
  def _differentiable_bounds(result: _BoundsResult) -> DifferentiableBounds:
    interval, exact, partials, hessian = result
    gradients = tuple(GradientInterval(source, lower, upper) for source, lower, upper, _ in partials)
    probability = None
    if exact is not None:
      weights = []
      for source, _, _, derivative in partials:
        if derivative is None:
          raise RuntimeError("Missing derivative for the exact query probability")
        weights.append((source, derivative))
      probability = _autograd_probability(exact, weights, hessian)
    return DifferentiableBounds(ProbabilityInterval(*interval), gradients, probability)

  def run_batch(
    self,
    inputs: Mapping[str, Sequence[Union[Tuple[List[Any], Optional[List[List[int]]]], List[Any]]]],
    output_relation: Union[str, List[str]],
    parallel: bool = False,
  ) -> List[ScallopCollection]:
    """
    Run the program in batch mode, returning one collection per task.
    Pass one output relation for all tasks or one name per task.
    """
    if not inputs:
      raise ValueError("Batch inputs must contain at least one relation")
    batch_size = len(next(iter(inputs.values())))
    output_relations = [output_relation] * batch_size if isinstance(output_relation, str) else output_relation
    if len(output_relations) != batch_size:
      raise ValueError(f"Expected {batch_size} output relations, got {len(output_relations)}")
    batch_inputs = {
      relation: [
        self._prepare_facts(relation, *task) if isinstance(task, tuple) else self._prepare_facts(relation, task)
        for task in tasks
      ]
      for relation, tasks in inputs.items()
    }
    results = self._internal.run_batch([[r] for r in output_relations], batch_inputs, parallel)
    return [ScallopCollection(self.provenance, coll) for (coll,) in results]

  @record_history
  def import_file(self, file_name: str):
    """
    Import a file given the file name
    """
    self._internal.import_file(file_name)

  @record_history
  def add_program(self, program: str):
    """
    Add a full scallop program string
    """
    self._internal.add_program(program)

  @record_history
  def add_item(self, item: str) -> List[str]:
    """
    Add an item and get the mentioned query relations
    """
    return self._internal.add_item(item)

  def load_stdlib(self):
    for ff in STDLIB["functions"]:
      self._internal.register_foreign_function(ff)
    for fp in STDLIB["predicates"]:
      self._internal.register_foreign_predicate(fp)
    for fa in STDLIB["attributes"]:
      self._internal.register_foreign_attribute(fa)

  @record_history
  def register_foreign_function(self, foreign_function: ForeignFunction):
    if type(foreign_function) == ForeignFunction:
      self._internal.register_foreign_function(foreign_function)
    else:
      raise Exception("Registering non-foreign-function. Consider decorating the function with @scallopy.foreign_function")

  @record_history
  def register_foreign_predicate(self, foreign_predicate: ForeignPredicate):
    if type(foreign_predicate) == ForeignPredicate:
      self._internal.register_foreign_predicate(foreign_predicate)
    else:
      raise Exception("Registering non-foreign-predicate. Consider decorating the function with @scallopy.foreign_predicate")

  @record_history
  def register_foreign_attribute(self, foreign_attribute: ForeignAttributeProcessor):
    if type(foreign_attribute) == ForeignAttributeProcessor:
      self._internal.register_foreign_attribute(foreign_attribute)
    else:
      raise Exception("Registering non-foreign-attribute. Consider decorating the function with @scallopy.attribute")

  def forward_function(
    self,
    output: Optional[str] = None,
    output_mapping: Optional[_OutputMapping] = None,
    output_mappings: Optional[Dict[str, Optional[_OutputMapping]]] = None,
    dispatch: str = "parallel",
    debug_provenance: bool = False,
    jit: bool = False,
    jit_name: str = "",
    recompile: bool = False,
    sparse_jacobian: bool = False,
  ) -> InternalScallopForwardFunction:
    """
    Generate a forward function for PyTorch module.

    Example:

    ``` python
    eval = ctx.forward_function("target")
    result = eval(input_1=tensor_1, input_2=tensor_2)
    ```

    :param output: the output relation name
    :param output_mapping: the output mapping for vectorization
    If not specified (i.e. None), will run until fixpoint
    """
    if jit or jit_name or recompile:
      raise ValueError(
        "Native JIT was removed. Remove jit, jit_name, and recompile (jit_recompile on "
        "ScallopForwardFunction) to use the RAM interpreter."
      )

    # Check PyTorch support
    if not torch_importer.has_pytorch():
      raise Exception("`forward_function` cannot be called when there is no PyTorch")

    from .forward import InternalScallopForwardFunction

    # Needs to be a differentiable context
    if "diff" in self.provenance or self.provenance == "custom": pass
    else: raise Exception("`forward_function` can only be called on context with differentiable provenance")

    # Forward function
    return InternalScallopForwardFunction(
      self,
      output,
      output_mapping,
      output_mappings,
      dispatch,
      debug_provenance,
      sparse_jacobian=sparse_jacobian,
    )

  def _refresh_training_eval_state(self, training):
    if self._train_k is not None or self._test_k is not None:
      if training: # Coming from nn.Module and `torch.train()` or `torch.eval()`
        if self._train_k is not None:
          self._internal.set_k(self._train_k)
        else:
          self._internal.set_k(self._k)
      else:
        if self._test_k is not None:
          self._internal.set_k(self._test_k)
        else:
          self._internal.set_k(self._k)

  @record_history
  def add_relation(
    self,
    relation_name: str,
    relation_types: Union[Tuple, type, str],
    input_mapping: Any = None,
    retain_topk: Optional[int] = None,
    non_probabilistic: bool = False,
    load_csv: Optional[Union[CSVFileOptions, str]] = None,
  ):
    """
    Add a relation to the context, where a relation is defined by its name
    and the tuple types.
    Idiomatic python types such as `str`, `int`, `bool` are supported, while
    internally being transformed into corresponding scallop types.
    (e.g. `int` -> `i32`).

    ``` python
    ctx.add_relation("edge", (int, int))
    ctx.add_relation("path", (int, int))
    ctx.add_relation("digit", int)
    ```

    Note that we usually accept a tuple. When a single element, such as `int`
    is provided, it will be internally transformed into a one-tuple.

    In addition to idiomatic python types, you can use the types provided in
    `types` module to gain access to native Scallop types:

    ``` python
    from scallopy import types
    ctx.add_relation("digit", (types.i8,))
    ```

    Of course, directly feeding type as string is ok too:

    ``` python
    ctx.add_relation("digit", "i8")
    ```

    You can load csv using this function

    ``` python
    ctx.add_relation(
      "edge", (int, int),
      load_csv = "FILE.csv",
    )
    ```

    If you wish to specify properties of CSV such as deliminator, you can
    use the `CSVFile` class

    ``` python
    edge_csv_file = CSVFile("FILE.csv", deliminator="\t")
    ctx.add_relation("edge", (int, int), load_csv=edge_csv_file)
    ```
    """

    # Helper function
    def _type_to_scallop_type_str(ty):
      if ty == str:
        return "String"
      elif ty == int:
        return "i32"
      elif ty == bool:
        return "bool"
      elif ty == float:
        return "f32"
      elif type(ty) == str:
        return ty
      elif isinstance(ty, TypeVar):
        return ty.__name__
      else:
        raise Exception(f"Unknown type `{ty}`")

    # Make sure that relation types is a tuple
    is_singleton_tuple = False
    if type(relation_types) == tuple:
      relation_types_tuple = relation_types
    elif type(relation_types) == type or type(relation_types) == str:
      is_singleton_tuple = True
      relation_types_tuple = (relation_types,)
    elif isinstance(relation_types, TypeVar):
      is_singleton_tuple = True
      relation_types_tuple = (relation_types.__name__,)
    else:
      raise Exception(f"Unknown relation types `{relation_types}`")

    # Create the decl str
    types_str = ", ".join([_type_to_scallop_type_str(ty) for ty in relation_types_tuple])
    relation_decl_str = f"{relation_name}({types_str})"

    # Invoke internal's add relation
    inserted_relation_name = self._internal.add_relation(relation_decl_str, load_csv=load_csv)

    # Sanity check
    assert relation_name == inserted_relation_name

    # Store the input mapping
    if input_mapping is not None:
      self.set_input_mapping(relation_name, input_mapping)

    # Store the retain topk property
    if retain_topk is not None:
      self._input_retain_topk[relation_name] = retain_topk

    # Store the non-probabilistic property
    if non_probabilistic:
      self._input_non_probabilistic[relation_name] = True

    # Store the input is singleton property
    if is_singleton_tuple:
      self._input_is_singleton[relation_name] = True

  @record_history
  def add_facts(
    self,
    relation: str,
    elems: Sequence[_FactT],
    disjunctions: Optional[List[List[int]]] = None,
  ) -> None:
    """
    Add facts to the relation under the context. The simple usage is as
    follows:

    ``` python
    ctx.add_facts("edge", [(0, 1), (1, 2)])
    ```

    Note that when adding relation, the relation name must be previously
    defined in the context using `add_relation` or by inferred from other
    rules. The function may throw error if the type does not match.

    When the context is associated with a non-unit provenance context,
    say "minmaxprob", one would need to provide the tag associated with
    each tuple. For "minmaxprob", as an example, one would need to invoke
    the function like this:

    ``` python
    ctx.add_facts("digit", [
      (0.90, (0, 1)),
      (0.01, (0, 2)),
      (0.03, (0, 3)),
    ])
    ```

    Note that there is a probability in the beginning and each tuple
    is now nested.

    :param relation: the name of the relation
    :param elems: the tuple elements
    :param disjunctions: the disjunctions
    """

    self._internal.add_facts(relation, self._prepare_facts(relation, elems, disjunctions))

  def add_untagged_facts(
    self,
    relation: str,
    elems: Sequence[_Fact],
    disjunctions: Optional[List[List[int]]] = None,
  ) -> None:
    """Add untagged facts to a relation that accepts them."""
    if self.requires_tag() and not self.relation_is_non_probabilistic(relation):
      raise ValueError(f"Relation {relation!r} requires tagged facts; use add_facts instead")
    self.add_facts(relation, cast(Sequence[_FactT], elems), disjunctions)

  @record_history
  def add_joint_choice(
    self, outcomes: Sequence[Tuple[Union[float, Tensor], Mapping[str, Sequence[_Fact]]]],
  ) -> None:
    """Add a categorical draw whose outcomes each assert a set of facts together."""
    self._internal.add_joint_choice([
      (weight, {
        relation: [
          (row,) if self._input_is_singleton.get(relation, False) and not isinstance(row, tuple) else row
          for row in rows
        ]
        for relation, rows in relations.items()
      })
      for weight, relations in outcomes
    ])

  def _prepare_facts(
    self,
    relation: str,
    elems: Sequence[Any],
    disjunctions: Optional[List[List[int]]] = None,
  ) -> List[Tuple[Any, ...]]:
    # First normalize the probability format
    if self.relation_is_non_probabilistic(relation) and self.requires_tag():
      elems = [(None, t) for t in elems]

    # Then normalize the singleton tuple format
    if relation in self._input_is_singleton and self._input_is_singleton[relation]:
      if self.requires_tag():
        elems = [(tag, (tup,)) if type(tup) != tuple else (tag, tup) for (tag, tup) in elems]
      else:
        elems = [(tup,) if type(tup) != tuple else tup for tup in elems]

    return self._process_disjunctive_elements(elems, disjunctions)

  @record_history
  def add_rule(self, rule: str, tag: Optional[Any] = None, demand: Optional[str] = None):
    """
    Add rule to the context. The rule will be compiled and compilation
    error may be raised.

    ``` python
    ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")
    ```

    In case non-unit provenance is used, a rule can be associated with
    tag. For example, in a probabilistic provenance context, one can
    associate probability with the rule.

    ``` python
    ctx.add_rule("born_in(a, \"china\") :- speaks(a, \"chinese\")", tag = 0.8)
    ```

    :param rule: a rule in scallop syntax
    :param tag: the tag associated with the rule
    """
    self._internal.add_rule(rule, tag=tag, demand=demand)

  @record_history
  def compile(self):
    self._internal.compile()

  def dump_front_ir(self):
    """
    Dump the Scallop front internal representation of the program compiled
    inside this context.
    """
    self._internal.dump_front_ir()

  def get_front_ir(self):
    """
    Get the Scallop front internal representation of the program in a `str` format
    """
    return self._internal.get_front_ir()

  def relation(self, relation: str) -> ScallopCollection:
    """
    Inspect the (computed) relation in the context. Will return
    a `ScallopCollection` which is iterable.

    ``` python
    for tuple in ctx.relation("edge"):
      print(tuple)
    ```

    :param relation: the name of the relation
    """
    int_col = self._internal.relation(relation)
    return ScallopCollection(self.provenance, int_col)

  def relation_rows(self, relation: str) -> Iterator[Row[_ValueT]]:
    """Iterate through computed tuples and their recovered provenance values."""
    return self.relation(relation)._rows()

  def has_relation(self, relation: str) -> bool:
    """
    Check if the compiled program contains a given relation.
    """
    return self._internal.has_relation(relation)

  def relation_is_computed(self, relation: str) -> bool:
    """
    Check if the relation is computed.
    """
    return self._internal.relation_is_computed(relation)

  def num_relations(self, include_hidden: bool = False) -> int:
    """
    Get the number of relations in the context.
    """
    return self._internal.num_relations(include_hidden=include_hidden)

  def relations(self, include_hidden: bool = False) -> List[str]:
    """
    Get a list of user defined relations in the context.
    If `include_hidden` is specified to be `True`, will also return the auxilliary
    relations (such as sub-formula, permutation, etc) in the list.
    """
    return self._internal.relations(include_hidden=include_hidden)

  def relation_field_names(self, relation: str) -> Optional[List[Optional[str]]]:
    """
    Get the field names of the relation, or None if it is not declared.
    Given the following Scallop program

    ``` scl
    type edge(from: i32, to: i32)
    ```

    We could get the field names as follows

    ``` py
    ctx.relation_field_names("edge") // ["from", "to"]
    ```
    """
    return self._internal.relation_field_names(relation)

  def set_non_probabilistic(self, relation: Union[str, List[str]], non_probabilistic: bool = True):
    """
    Set the relation to be non-probabilistic
    """
    if isinstance(relation, list):
      for r in relation:
        self._set_relation_non_probabilistic(r, non_probabilistic)
    else:
      self._set_relation_non_probabilistic(relation, non_probabilistic)

  def _set_relation_non_probabilistic(self, relation: str, non_probabilistic: bool):
    if self.has_relation(relation):
      self._input_non_probabilistic[relation] = non_probabilistic
    else:
      raise Exception(f"Unknown relation {relation}")

  def set_input_mapping(
    self,
    relation: str,
    input_mapping: Any,
    disjunctive: bool = False,
    disjunctive_dim: Optional[int] = None,
    retain_threshold: Optional[float] = None,
    retain_k: Optional[int] = None,
    sample_dim: Optional[int] = None,
    sample_strategy: Optional[Literal["top", "categorical"]] = "top",
  ):
    """
    Set the input mapping for the given relation
    """

    # Try to create an input mapping, pass all configurations
    mapping = InputMapping(
      input_mapping,
      disjunctive=disjunctive,
      disjunctive_dim=disjunctive_dim,
      retain_threshold=retain_threshold,
      retain_k=retain_k,
      sample_dim=sample_dim,
      sample_strategy=sample_strategy,
      supports_disjunctions=self.supports_disjunctions(),
    )

    # Check if the tuples in the input mapping matches
    for t in mapping.all_tuples():
      if not self._internal.check_tuple(relation, t):
        raise Exception(f"The tuple {t} in the input mapping does not match the type of the relation `{relation}`")

    # If there is no problem, set the input_mapping property
    self._input_mappings[relation] = mapping

  def has_input_mapping(self, relation: str) -> bool:
    if relation in self._input_mappings:
      if self._input_mappings[relation] is not None and self._input_mappings[relation].kind is not None:
        return True
    return False

  def set_sample_topk_facts(self, relation: str, amount: int):
    if relation in self._input_mappings:
      self._input_mappings[relation].set_sample_topk_facts(amount)
    else:
      raise Exception(f"Unknown relation {relation}")

  def requires_tag(self) -> bool:
    """
    Returns whether the context requires facts to be associated with tags
    """
    if self.provenance == "unit" or self.provenance == "proofs":
      return False
    else:
      return True

  def relation_is_non_probabilistic(self, relation: str) -> bool:
    if relation in self._input_non_probabilistic and self._input_non_probabilistic[relation]:
      return True
    else:
      return False

  def is_probabilistic(self) -> bool:
    """
    Check if the current context supports probabilistic reasoning
    """
    PROVENANCE_SUPPORTING_PROBABILITY = set([
      "anytimebounds",
      "probproofs",
      "topkproofs",
      "samplekproofs",
      "topbottomkclauses",
      "minmaxprob",
      "addmultprob",
    ])
    return self.provenance in PROVENANCE_SUPPORTING_PROBABILITY

  def supports_disjunctions(self) -> bool:
    """
    Check if the current context supports fact disjunctions
    """
    PROVENANCE_SUPPORTING_DISJUNCTIONS = set([
      "proofs",
      "anytimebounds",
      "topkproofs",
      "topbottomkclauses",
      "diffsamplekproofs",
      "difftopkproofs",
      "difftopbottomkclauses",
      "difftopkproofsdebug",
    ])
    return self.provenance in PROVENANCE_SUPPORTING_DISJUNCTIONS

  def _process_disjunctive_elements(
    self,
    elems: Sequence[Tuple[Any, ...]],
    disjunctions: Optional[List[List[int]]],
  ) -> List[Tuple[Any, ...]]:
    processed_elems = [e for e in elems]

    # Check if we the provenance supports handling disjunctions
    if self.supports_disjunctions():

      # Collect the facts occurred in disjunctions and those who don't
      visited_fact_ids = set()

      # If there are indeed disjunctions
      if disjunctions and len(disjunctions) > 0:
        # Go through each disjunction
        for disjunction in disjunctions:
          if self.provenance == "anytimebounds" and not disjunction:
            raise ValueError("An exclusion group must contain at least one fact")
          # Assign a new disjunction id for this one
          disjunction_id = self._mutual_exclusion_counter.get_and_increment()

          # Go through the facts and update the tag to include disjunction id
          for fact_id in disjunction:
            if self.provenance == "anytimebounds":
              if not 0 <= fact_id < len(elems):
                raise ValueError(f"Invalid exclusion fact index {fact_id}")
              if fact_id in visited_fact_ids:
                raise ValueError(f"Fact {fact_id} occurs more than once in exclusion groups")
              if elems[fact_id][0] is None:
                raise ValueError("An exclusion alternative must have a weight")
            visited_fact_ids.add(fact_id)
            if self.requires_tag():
              (tag, tup) = elems[fact_id]
              processed_elems[fact_id] = ((tag, disjunction_id), tup)
            else:
              processed_elems[fact_id] = (disjunction_id, elems[fact_id])

      # Update facts who is not inside of any disjunction
      for (fact_id, fact) in enumerate(elems):
        if fact_id not in visited_fact_ids:
          if self.requires_tag():
            (tag, tup) = fact
            if type(tag) == tuple:
              processed_elems[fact_id] = ((*tag, None), tup)
            else:
              processed_elems[fact_id] = ((tag, None), tup)
          else:
            processed_elems[fact_id] = (None, fact)

    # Return the processed elements
    return processed_elems
