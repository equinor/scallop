from typing import Dict, Union, List, Optional, Tuple, Any, TypeAlias
import logging

from .context import ScallopContext
from .modes import ProvenanceMode, from_name, _warn_string_provenance
from .provenance import ScallopProvenance
from .utils import _mapping_tuple, get_backward_proxy

from . import torch_importer

_OutputMapping: TypeAlias = Union[List[Any], Tuple[Any, ...], range]


class ScallopForwardFunction(torch_importer.Module):
  def __init__(
    self,
    file: Optional[str] = None,
    program: Optional[str] = None,
    relations: Optional[List[Tuple[str, Tuple]]] = None,
    rules: Optional[List[str]] = None,
    facts: Optional[Dict[str, List]] = None,
    provenance: Union[str, ProvenanceMode[Any, Any], None] = None,
    custom_provenance: Optional[ScallopProvenance] = None,
    non_probabilistic: Optional[List[str]] = None,
    input_mappings: Optional[Dict[str, Any]] = None,
    output_relation: Optional[str] = None,
    output_mapping: Optional[_OutputMapping] = None,
    output_mappings: Optional[Dict[str, Optional[_OutputMapping]]] = None,
    k: Optional[int] = None,
    train_k: Optional[int] = None,
    test_k: Optional[int] = None,
    *,
    early_discard: Optional[bool] = None,
    iter_limit: Optional[int] = None,
    retain_topk: Optional[Dict[str, int]] = None,
    dispatch: str = "parallel",
    sparse_jacobian: bool = False,
    monitors: List[str] = [],
  ):
    super(ScallopForwardFunction, self).__init__()

    if isinstance(provenance, ProvenanceMode):
      if custom_provenance is not None:
        raise TypeError("Specify custom provenance on the provenance mode")
    else:
      if provenance is not None:
        _warn_string_provenance()
      provenance = from_name(
        "difftopkproofs" if provenance is None else provenance,
        custom_provenance=custom_provenance,
      )

    self.ctx = ScallopContext(
      provenance=provenance,
      k=k,
      train_k=train_k,
      test_k=test_k,
      monitors=monitors,
    )

    # Import the file if specified
    if file is not None:
      self.ctx.import_file(file)

    # Add a full program if specified
    if program is not None:
      self.ctx.add_program(program)

    # Add relation if specified
    if relations is not None:
      for (relation_name, relation_type) in relations:
        self.ctx.add_relation(relation_name, relation_type)

    # Add rules if specified
    if rules is not None:
      for rule in rules:
        self.ctx.add_rule(rule)

    # Set the non-probabilistic property if specified
    if non_probabilistic is not None:
      self.ctx.set_non_probabilistic(non_probabilistic)

    # Set the input mappings if specified
    if input_mappings is not None:
      for (relation, mapping) in input_mappings.items():
        self.ctx.set_input_mapping(relation, mapping)

    # Set the retain top-k
    if retain_topk is not None:
      for (relation, k) in retain_topk.items():
        self.ctx.set_sample_topk_facts(relation, k)

    # Add input facts if specified
    if facts is not None:
      for (relation, elems) in facts.items():
        self.ctx.add_facts(relation, elems)

    # Configurations: iteration limit
    if iter_limit is not None:
      self.ctx.set_iter_limit(iter_limit)

    # Configurations: early discarding
    if early_discard is not None:
      self.ctx.set_early_discard(early_discard)

    # Create the forward function
    self.forward_fn = self.ctx.forward_function(
      output=output_relation,
      output_mapping=output_mapping,
      output_mappings=output_mappings,
      dispatch=dispatch,
      sparse_jacobian=sparse_jacobian,
    )

  def forward(self, *pos_args, **kw_args):
    return self.forward_fn(*pos_args, **kw_args)


class InternalScallopForwardFunction(torch_importer.Module):
  """
  A Scallop PyTorch forward function.

  :param dispatch str, can be chosen from `"single"`, `"serial"`, or `"parallel"`
  """
  def __init__(
    self,
    ctx: ScallopContext,
    output: Optional[str] = None,
    output_mapping: Optional[_OutputMapping] = None,
    output_mappings: Optional[Dict[str, Optional[_OutputMapping]]] = None,
    dispatch: str = "parallel",
    debug_provenance: bool = False,
    sparse_jacobian: bool = False,
  ):
    super(InternalScallopForwardFunction, self).__init__()

    # Parameters
    self.ctx = ctx
    self.dispatch = dispatch
    self.debug_provenance = debug_provenance
    self.sparse_jacobian = sparse_jacobian

    # Preprocess the dispatch
    if self.ctx.provenance == "custom" and self.dispatch == "parallel":
      logging.warning("custom provenance does not support parallel dispatch; falling back to serial dispatch. Consider creating the forward function using `dispatch=\"serial\"`.")
      self.dispatch = "serial"

    # Populate the output and output mapping fields
    self._process_output_mapping(output, output_mapping, output_mappings)
    for output_relation in self.outputs:
      if not self.ctx.has_relation(output_relation):
        raise Exception(f"Unknown relation `{output_relation}`; cannot construct forward function")

    # Default tensor apply function is an identity function; later it might be
    # a `to("cpu")` or `to("gpu")` function.
    self._torch_tensor_apply = lambda x: x

  def __getstate__(self):
    """
    Serialize into pickle state
    """
    state = self.__dict__.copy()
    del state["_torch_tensor_apply"]
    return state

  def __setstate__(self, state):
    """
    Deserialize from pickle state
    """
    if state.get("jit"):
      raise ValueError("Native JIT was removed. Recreate this forward function without jit=True.")
    self.__dict__.update(state)
    self._torch_tensor_apply = lambda x: x

  def _process_output_mapping(self, output, output_mapping, output_mappings):
    """
    Populate the following fields
    - self.outputs: List[str]
    - self.output_mappings: Dict[str, Tuple[bool, List[Tuple]]]
      - Mapping from relation name to (single_element, mappings) tuple
    """
    # Initialize
    self.output_mappings = {}

    # Check if output field is provided
    if output:
      self.outputs = [output] # Then there will be only one output
      self.output_mappings[output] = self._process_one_output_mapping(output_mapping)
    elif output_mappings and type(output_mappings) == dict:
      self.outputs = [r for (r, _) in output_mappings.items()] # There will be many outputs
      for (r, m) in output_mappings.items(): # Process each item in output mappings
        self.output_mappings[r] = self._process_one_output_mapping(m)
    else:
      self.outputs = [] # Then there will be no outputs

    # Check the types of the output mappings
    for (relation, info) in self.output_mappings.items():
      if info is not None:
        (_, output_mapping) = info
        for tuple in output_mapping:
          if not self.ctx._internal.check_tuple(relation, tuple):
            raise Exception(f"The tuple {tuple} in the output mapping does not match the type of the relation `{relation}`")

  def _process_one_output_mapping(self, output_mapping):
    if type(output_mapping) == list:
      return (False, [_mapping_tuple(t) for t in output_mapping])
    elif type(output_mapping) == tuple:
      return (True, [_mapping_tuple(output_mapping)])
    elif type(output_mapping) == range:
      return self._process_one_output_mapping(list(output_mapping))
    elif output_mapping is None:
      return None
    else:
      raise Exception(f"Unknown output mapping type `{type(output_mapping)}`")

  def _apply(self, fn, recurse=True):
    """
    Overriding `nn.Module`'s _apply function to accept additional tensor apply
    function `f`. Usually this `f` function is a `.to("cpu")` or `.to("gpu")`
    function. For Scallop to support GPU, we will use this function `f` to
    apply to any tensor that we generate in the computation process.
    """
    self._torch_tensor_apply = fn
    return super()._apply(fn, recurse=recurse)

  def forward(
    self,
    disjunctions: Dict[str, List[List[List[int]]]] = {},
    output_relations: Optional[List[Union[str, List[str]]]] = None,
    output_mappings: Optional[Union[List, Dict[str, List]]] = None,
    **input_facts: Union[torch_importer.Tensor, List],
  ):
    """
    Invoke the forward function with the given facts

    The `facts` and `disjunctions` need to be batched; and we assume the batch size
    to be B. For built-in differentiable provenances, input tensor tags must
    have the same dtype within a call.

    `output_relations` can be one of the following format
    - None, if outputs are provided when constructing the ForwardFunction
    - [rela 1, rela 2, ..., rela B], if we want each data-point to produce different relations
    """
    self.ctx._refresh_training_eval_state(self.training) # Set train/eval
    self.ctx._internal.set_non_incremental()
    self.ctx._internal.compile()

    # First make sure that all facts share the same batch size
    batch_size = self._compute_and_check_batch_size(input_facts)

    # Process the input into a unified form
    all_inputs = self._process_all_input_facts(batch_size, input_facts, disjunctions)

    # Process the output into a list of output relations
    if output_relations is None:
      current_output_relations = [self.outputs] * batch_size
    elif type(output_relations) == str:
      current_output_relations = [[output_relations]] * batch_size
    elif type(output_relations) == list:
      current_output_relations = [rs if type(rs) == list else [rs] for rs in output_relations]

    # Making sure that output relations are well-formed
    if any([len(rs) == 0 for rs in current_output_relations]):
      raise Exception("There exists a 0 output relations data-point")
    if len(current_output_relations) != batch_size:
      raise Exception(f"Number of output relations ({len(current_output_relations)}) does not match the batch size ({batch_size})")

    # Process the output_mappings
    # the set of output relations for the first data-point in the batch
    first_output_relations = current_output_relations[0]
    if output_mappings is not None:
      # First initialize the `current_output_mappings` to be a deep copy of the existing output mapping
      # Later on if there is new temporary output mapping provided that is specific to the batch, we will
      # use the temporary output mapping to overwrite the mappings in the existing output mapping
      current_output_mappings = {k: v for (k, v) in self.output_mappings.items()}

      # Compute some statistics of output relations to make sure that the input is well-formed
      set_of_output_relations = set([r for rs in current_output_relations for r in rs])
      num_output_relations = len(set_of_output_relations)

      # Check if there is only one single output relation
      if num_output_relations == 1:
        output_relation = first_output_relations[0]
        if type(output_mappings) == list:
          # the output mappings could be a single list, which would by default be the output mapping
          # for the only output relation that is specified
          assert len(output_mappings) > 0, "Expect the `output_mappings` to be non-empty"
          current_output_mappings[output_relation] = self._process_one_output_mapping(output_mappings)
        elif isinstance(output_mappings, dict):
          # the output_mappings struct could be a dictionary. In this case it could either be empty or contains
          # exactly one output mapping which is for the only output relation.
          for (rela_name, rela_output_mapping) in output_mappings.items():
            assert rela_name in set_of_output_relations, f"The provided output mapping {rela_name} is not among the set of output relations"
            assert len(rela_output_mapping) > 0, f"Expect the output mapping for the `{rela_name}` relation to be non-empty"
            current_output_mappings[rela_name] = self._process_one_output_mapping(rela_output_mapping)
        else:
          assert False, "Unknown format of `output_mappings` when calling Scallop forward. Expecting list or dict."
      else:
        # we make sure that this should be the same for every single data-point of the rest of the batch
        is_uniform_output = all([rs == first_output_relations for rs in current_output_relations[1:]])
        assert is_uniform_output, "We expect that the output relations to be the same across all datapoints in a batch"

        # we make sure that the output_mappings is provided for all
        assert isinstance(output_mappings, dict), "Expect the `output_mappings` variable to be a dict for the batch with multiple output relations"
        for (rela_name, rela_output_mapping) in output_mappings.items():
          assert rela_name in set_of_output_relations, f"The provided output mapping {rela_name} is not among the set of output relations"
          assert len(rela_output_mapping) > 0, f"Expect the output mapping for the `{rela_name}` relation to be non-empty"
          current_output_mappings[rela_name] = self._process_one_output_mapping(rela_output_mapping)
    else:
      current_output_mappings = self.output_mappings

    # Check task dispatcher
    if self.dispatch == "single":
      # Execute scallop program for each task from python
      input_tags = []
      output_results = []
      for task_id in range(batch_size):
        (task_input_tags, task_output_results) = self._run_single(task_id, all_inputs, current_output_relations[task_id], current_output_mappings)
        input_tags.append(task_input_tags)
        output_results.append(task_output_results)
    elif self.dispatch == "serial":
      # Directly dispatch all the inputs to rust
      (input_tags, output_results) = self._run_batch(batch_size, all_inputs, current_output_relations, current_output_mappings, parallel=False)
    elif self.dispatch == "parallel":
      # Dispatch all the inputs to rust and call rayon as parallelism backend
      (input_tags, output_results) = self._run_batch(batch_size, all_inputs, current_output_relations, current_output_mappings, parallel=True)
    else:
      raise Exception(f"Unknown dispatch type `{self.dispatch}`")

    # Process the output
    return self._process_output(batch_size, input_tags, output_results, current_output_mappings)

  def _compute_and_check_batch_size(self, inputs: Dict[str, Union[torch_importer.Tensor, List]]) -> int:
    """
    Given the inputs, check if the batch size is consistent over all relations.
    If so, return the batch size.
    Otherwise, when there is no input or any batch size is inconsistent, throw exception.
    """
    batch_size = None
    for (rela, rela_facts) in inputs.items():
      if batch_size is None:
        batch_size = len(rela_facts)
      elif batch_size != len(rela_facts):
        raise Exception(f"Inconsistency in batch size: expected {batch_size}, found {len(rela_facts)} for relation `{rela}`")
    if batch_size is None:
      raise Exception("There is no input to the forward function")
    else:
      return batch_size

  def _process_all_input_facts(self, batch_size, all_input_facts, all_disjunctions):
    """
    Given all the input facts where facts may be lists or tensors,
    process them into a unified form.
    """
    processed_inputs = {}
    check_dtype = self.ctx.provenance == "diffminmaxprob" or self._has_output_hook()
    tensor_type = torch_importer.torch.Tensor if check_dtype else None
    supports_disjunctions = check_dtype and self.ctx.supports_disjunctions()
    dtype = None
    for (rela, rela_facts) in all_input_facts.items():
      processed_inputs[rela] = []
      for task_id in range(batch_size):
        ds = all_disjunctions[rela][task_id] if all_disjunctions is not None and rela in all_disjunctions else None
        facts = self._process_input_facts(rela, rela_facts[task_id], ds)
        if tensor_type is not None:
          for tag, _ in facts:
            if supports_disjunctions and isinstance(tag, tuple):
              tag = tag[0]
            if isinstance(tag, tensor_type):
              if dtype is None:
                dtype = tag.dtype
              elif tag.dtype != dtype:
                raise TypeError(
                  f"Expected forward input tags with dtype {dtype}, got {tag.dtype} "
                  f"in relation {rela!r} at batch index {task_id}"
                )
        processed_inputs[rela].append(facts)
    return processed_inputs

  def _process_input_facts(self, rela, rela_facts, disjunctions) -> List[Tuple]:
    """
    Given input facts of one single relation (and its disjunctions), process it into
    a unified form of input facts along with its disjunctions.

    Note that the disjunction IDs will be normalized as well.
    """
    if rela in self.ctx._input_non_probabilistic and self.ctx._input_non_probabilistic[rela]:
      # Add non-probabilistic facts; there will be no disjunctions
      return [(None, f) for f in rela_facts]
    else:
      # Process the facts
      ty = type(rela_facts) # The type of relation facts
      index_mapping = None # The index mapping of given facts and preprocessed facts if there is removal of facts

      # If the facts are directly provided as list
      if ty == list:
        facts = rela_facts
        if rela in self.ctx._input_retain_topk:
          k = min(self.ctx._input_retain_topk[rela], len(facts))
          if disjunctions is not None:
            indexed_facts = sorted(enumerate(facts), key=lambda x: x[1][0].item(), reverse=True)[:k]
            index_mapping = {j: i for (i, (j, _)) in enumerate(indexed_facts)}
            facts = [f for (_, f) in indexed_facts]
          else:
            facts = sorted(facts, key=lambda x: x[0].item(), reverse=True)[:k]

        # Remap disjunction
        remapped_disjs = [[index_mapping[i] for i in d if i in index_mapping] for d in disjunctions] if index_mapping is not None else disjunctions

        # Process elements with this disjunction
        facts = self.ctx._process_disjunctive_elements(facts, remapped_disjs)

      # If the facts are provided as Tensor
      elif ty == torch_importer.torch.Tensor:
        if self._has_debug_info():
          raise Exception(f"scallopy.forward with debug provenance `{self.ctx.provenance}` does not accept tensor inputs. Consider passing lists instead")

        if rela not in self.ctx._input_mappings:
          raise Exception(f"scallopy.forward receives vectorized Tensor input. However there is no `input_mapping` provided for relation `{rela}`")

        # Use the input mapping to process
        facts = self.ctx._input_mappings[rela].process_tensor(rela_facts)
      else:
        raise Exception(f"Unknown input facts type. Expected Tensor or List, found {ty}")

      # Add the facts
      return facts

  def _run_single(self, task_id, all_inputs, output_relations, output_mappings):
    """
    Run a single task (identified by `task_id`)

    Internally we will pick out the inputs corresponding to this task only.
    The task is manually dispatched using internal Scallopy python API.
    """

    # Clone a context for scallop execution
    temp_ctx = self.ctx.clone()

    # Add the facts into the context
    for (rela, rela_inputs) in all_inputs.items():
      if not self.ctx.has_relation(rela):
        raise Exception(f"Unknown relation `{rela}`")
      facts = rela_inputs[task_id]
      temp_ctx._internal.add_facts(rela, facts)

    # Execute the context
    if self.debug_provenance:
      temp_ctx._internal.run_with_debug_tag()
    else:
      temp_ctx.run()

    # Get input tags
    input_tags = temp_ctx._internal.input_tags()

    # Get the internal collection for the target output
    if self.debug_provenance: cs = [temp_ctx._internal.relation_with_debug_tag(r) for r in output_relations]
    else: cs = [temp_ctx._internal.relation(r) for r in output_relations]

    # Process the collection to get the output results
    output_results = [self._process_single_output(output_relations[i], c, output_mappings) for (i, c) in enumerate(cs)]

    # Return
    return (input_tags, output_results)

  def _run_batch(self, batch_size, all_inputs, output_relations, output_mappings, parallel: bool):
    """
    Run a batch of tasks
    """
    results = self.ctx._internal.run_batch(output_relations, all_inputs, parallel)
    input_tags, output_results = [], []
    for task_id in range(batch_size):
      input_tags.append(results[task_id][0].input_tags())
      output_results.append([self._process_single_output(output_relations[task_id][i], c, output_mappings) for (i, c) in enumerate(results[task_id])])
    return (input_tags, output_results)

  def _process_single_output(self, relation_name, internal_collection, output_mappings):
    """
    Given a raw output collection from internal Scallop module, process the output with a given output mapping
    """
    internal_result_dict = { tup: tag for (tag, tup) in internal_collection }
    if relation_name in output_mappings and output_mappings[relation_name] is not None:
      return [internal_result_dict[t] if t in internal_result_dict else None for t in output_mappings[relation_name][1]]
    else:
      return internal_result_dict

  def _process_output(self, batch_size, input_tags, output_results, output_mappings):
    """
    Given all the outputs from internal Scallop module, process the outputs
    """

    # First make sure that the outputs are well-defined
    if len(self.outputs) == 0:
      outputs = list(output_mappings.keys())
    else:
      outputs = self.outputs
    assert type(outputs) == list, "Expect outputs to be a list"
    assert len(outputs) > 0, "Expect non-empty output from a forward function; however the current `outputs` is empty"
    assert all(type(output_rel) == str for output_rel in outputs), "Expect each element of output array to be a string (relation name)"

    # Then depending on how many of the output relations, process the output for each output relation
    if len(outputs) == 1:
      return self._process_one_output_wrapper(0, outputs[0], batch_size, input_tags, output_results, output_mappings)
    elif len(outputs) > 1:
      return {rel_name: self._process_one_output_wrapper(i, rel_name, batch_size, input_tags, output_results, output_mappings) for (i, rel_name) in enumerate(outputs)}

  def _process_one_output_wrapper(self, rel_index, rel_name, batch_size, input_tags, output_results, output_mappings):
    if output_mappings[rel_name] is not None:
      (single_element, output_mapping) = output_mappings[rel_name]
      return self._process_one_output(batch_size, input_tags, [r[rel_index] for r in output_results], single_element, output_mapping)
    else:
      return self._process_one_output(batch_size, input_tags, [r[rel_index] for r in output_results], False, None)

  def _process_one_output(self, batch_size, input_tags, output_results, single_element: bool, output_mapping: Optional[List[Tuple]]):
    output_parts = []

    # If there is no given output mapping, try
    if output_mapping is not None:
      # Integrate the outputs
      v = self._batched_prob(output_results)
      if self._has_output_hook() and v.requires_grad:
        v = self._batched_proxy_output(v, input_tags, output_results)
      v = v.view(-1) if single_element else v

      # Return the output
      output_parts.append(v)

      # If has debug information
      if self._has_debug_info():
        output_parts.append(self._batched_debug_info(output_results))
    else:
      # Collect all possible outputs, make them into a list
      possible_outputs = set()
      for internal_output_result in output_results:
        for (o, _) in internal_output_result.items():
          possible_outputs.add(o)
      post_output_mapping = list(possible_outputs)

      # Make the result into a batch
      post_output_results = []
      for internal_result_dict in output_results:
        post_output_results.append([internal_result_dict[t] if t in internal_result_dict else None for t in post_output_mapping])

      # Check if there is no output result
      if len(possible_outputs) == 0:
        # Return empty mapping and empty tensor
        output_parts.append([])
        output_parts.append(torch_importer.torch.zeros(batch_size, 0, requires_grad=self.training))

        # If has debug information
        if self._has_debug_info():
          output_parts.append([[] for _ in range(batch_size)])
      else:
        # Integrate the outputs based on all the output results
        v = self._batched_prob(post_output_results)
        if self._has_output_hook() and v.requires_grad:
          # Original: v.register_hook(self._batched_output_hook(input_tags, post_output_results))
          v = self._batched_proxy_output(v, input_tags, post_output_results)
        v = v.view(-1) if single_element else v

        # Return
        output_parts.append(post_output_mapping)
        output_parts.append(v)

        # If has debug information
        if self._has_debug_info():
          output_parts.append(self._batched_debug_info(post_output_results))

    if len(output_parts) == 1:
      return output_parts[0]
    else:
      return tuple(output_parts)

  def _batched_prob(
    self,
    tasks: List[List[Any]],
  ) -> torch_importer.Tensor:
    # Provenance diffminmaxprob
    if self.ctx.provenance == "diffminmaxprob":
      tensor_results = []
      for task_results in tasks:
        task_tensor_results = []
        for task_tup_result in task_results:
          if task_tup_result is not None:
            (s, p) = task_tup_result
            if s == 1:
              task_tensor_results.append(p)
            elif s == 0:
              task_tensor_results.append(self._torch_tensor_apply(torch_importer.torch.tensor(p, requires_grad=True)))
            else:
              task_tensor_results.append(1 - p)
          else:
            task_tensor_results.append(self._torch_tensor_apply(torch_importer.torch.tensor(0.0, requires_grad=True)))
        tensor_results.append(torch_importer.torch.stack(task_tensor_results))
      return torch_importer.torch.stack(tensor_results)

    # Provenance diff addmultprob / proofs
    # -- the provenances that returns a full derivatives array associated with the probability
    elif self.ctx.provenance == "diffaddmultprob" or \
         self.ctx.provenance == "diffnandmultprob" or \
         self.ctx.provenance == "diffmaxmultprob" or \
         self.ctx.provenance == "diffnandminprob" or \
         self.ctx.provenance == "difftopkproofs" or \
         self.ctx.provenance == "diffsamplekproofs" or \
         self.ctx.provenance == "difftopbottomkclauses" or \
         self.ctx.provenance == "difftopkproofsdebug":
      tensor_results = []
      for task_results in tasks:
        task_tensor_results = []
        for task_tup_result in task_results:
          if task_tup_result is not None:
            p = task_tup_result[0] # The 0-th element of the differentiable result is always a single probability
            task_tensor_results.append(torch_importer.torch.tensor(p, requires_grad=True))
          else:
            task_tensor_results.append(torch_importer.torch.tensor(0.0, requires_grad=True))
        tensor_results.append(torch_importer.torch.stack(task_tensor_results))
      return self._torch_tensor_apply(torch_importer.torch.stack(tensor_results))

    # If we have a custom provenance, use its collate function
    elif self.ctx.provenance == "custom":
      provenance = self.ctx._custom_provenance
      if provenance is None:
        raise RuntimeError("Missing custom provenance")
      return provenance.collate(tasks)

    # Non-differentiable provenance
    else:
      raise Exception("[Internal Error] Should not happen")

  def _has_output_hook(self):
    if self.ctx.provenance == "diffaddmultprob": return True
    elif self.ctx.provenance == "diffnandmultprob": return True
    elif self.ctx.provenance == "diffmaxmultprob": return True
    elif self.ctx.provenance == "diffnandminprob": return True
    elif self.ctx.provenance == "diffsamplekproofs": return True
    elif self.ctx.provenance == "difftopkproofs": return True
    elif self.ctx.provenance == "difftopbottomkclauses": return True
    elif self.ctx.provenance == "difftopkproofsdebug": return True
    else: return False

  def _has_debug_info(self):
    if self.ctx.provenance == "difftopkproofsdebug": return True
    else: return False

  def _batched_proxy_output(
    self,
    output_tensor: torch_importer.Tensor,
    input_tags: List[Optional[List[Any]]],
    tasks: List[List[Any]],
  ) -> torch_importer.Tensor:
    if self.ctx.provenance == "diffminmaxprob":
      return output_tensor
    elif self.ctx.provenance == "diffaddmultprob" or \
         self.ctx.provenance == "diffnandmultprob" or \
         self.ctx.provenance == "diffmaxmultprob" or \
         self.ctx.provenance == "diffnandminprob" or \
         self.ctx.provenance == "difftopkproofs" or \
         self.ctx.provenance == "diffsamplekproofs" or \
         self.ctx.provenance == "difftopbottomkclauses" or \
         self.ctx.provenance == "difftopkproofsdebug":
      proof_input_tags = []
      for tags in input_tags:
        if tags is None:
          raise RuntimeError(f"Missing input tags for {self.ctx.provenance}")
        proof_input_tags.append(tags)
      return self._diff_proofs_batched_proxy_output(output_tensor, proof_input_tags, tasks)
    else:
      raise Exception("[Internal Error] Should not happen")

  def _diff_proofs_batched_proxy_output(
    self,
    output_tensor: torch_importer.Tensor,
    input_tags: List[List[Any]],
    output_batch: List[List[Any]],
  ) -> torch_importer.Tensor:
    # Prepare the dimensions
    batch_size = len(output_batch)
    num_inputs = max(len(ts) for ts in input_tags)
    num_outputs = len(output_batch[0])

    # Check if there is no input
    if num_inputs == 0:
      return output_tensor

    torch = torch_importer.torch
    flat_tags = [tag for tags in input_tags for tag in tags]
    reference = None
    for tag in flat_tags:
      if tag is None:
        continue
      if not isinstance(tag, torch.Tensor):
        raise TypeError(f"Expected a tensor proof tag, got {type(tag).__name__}")
      if reference is None:
        reference = tag
      elif tag.dtype != reference.dtype:
        raise TypeError(f"Expected proof tags with dtype {reference.dtype}, got {tag.dtype}")
    zero = reference.new_zeros(()) if reference is not None else self._torch_tensor_apply(torch.tensor(0.0))
    packed = torch.stack([zero if tag is None else tag for tag in flat_tags])
    rows = packed.split([len(tags) for tags in input_tags])
    mat_i = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)

    # Check whether we want to use sparse jacobian
    if self.sparse_jacobian:
      # Populate the indices and values to later construct the sparse matrix
      indices, values = [], []
      for batch_id, task_result in enumerate(output_batch): # batch_size
        for output_id, output_tag in enumerate(task_result): # output_size
          if output_tag is not None:
            deriv = output_tag[1] # The 1st element of the differentiable result is always the derivative
            for (input_id, weight) in deriv:
              indices.append([batch_id, output_id, input_id])
              values.append(weight)

      # Convert indices and values into tensors
      indices = torch_importer.torch.tensor(indices, dtype=torch_importer.torch.long).reshape(-1, 3).t()
      values = torch_importer.torch.tensor(values)

      # Create the sparse tensor
      mat_w = self._torch_tensor_apply( # making sure that the tensor is on the intended device
        torch_importer.torch.sparse_coo_tensor(
          indices, values, (batch_size, num_outputs, num_inputs), check_invariants=True))
    else:
      # mat_w: Weight matrix
      mat_w = self._torch_tensor_apply(torch_importer.torch.zeros(batch_size, num_outputs, num_inputs))
      for (batch_id, task_result) in enumerate(output_batch): # batch_size
        for (output_id, output_tag) in enumerate(task_result): # output_size
          if output_tag is not None:
            deriv = output_tag[1] # The 1-st element of the differentiable result is always the derivative
            for (input_id, weight) in deriv:
              mat_w[batch_id, output_id, input_id] = weight

    # backward hook
    BackwardProxy = get_backward_proxy()
    proxied_output = BackwardProxy.apply(
      mat_i,
      output_tensor,
      mat_w,
      self.sparse_jacobian)
    return proxied_output

  def _batched_debug_info(
    self,
    output_results: List[List[Any]],
  ) -> List[List[Any]]:
    if self.ctx.provenance == "difftopkproofsdebug":
      results = []
      for task_results in output_results:
        task_tensor_results = []
        for task_tup_result in task_results:
          if task_tup_result is not None:
            proofs = task_tup_result[2] # The 2-th element of the difftopkproofsdebug result is the proofs
            task_tensor_results.append(proofs)
          else:
            task_tensor_results.append([])
        results.append(task_tensor_results)
      return results
    raise RuntimeError(f"No debug information for {self.ctx.provenance}")
