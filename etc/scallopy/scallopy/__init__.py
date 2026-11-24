from typing import TYPE_CHECKING as _TYPE_CHECKING

from .attribute import foreign_attribute as foreign_attribute, ForeignAttributeProcessor as ForeignAttributeProcessor
from .context import ScallopContext as ScallopContext
from .provenance import ScallopProvenance as ScallopProvenance
from .function import GenericTypeParameter as GenericTypeParameter, foreign_function as foreign_function, ForeignFunction as ForeignFunction
from .predicate import Facts as Facts, foreign_predicate as foreign_predicate, ForeignPredicate as ForeignPredicate
from .value_types import *  # noqa: F403
from .tag_types import *  # noqa: F403
from .input_mapping import InputMapping as InputMapping
from .plugin import PluginRegistry as PluginRegistry, Plugin as Plugin
from .syntax import *  # noqa: F403
from .scallopy import torch_tensor_enabled as torch_tensor_enabled

from . import input_output as input_output

io = input_output

if _TYPE_CHECKING:
  from .forward import ScallopForwardFunction as ScallopForwardFunction

  ForwardFunction = ScallopForwardFunction
  Module = ScallopForwardFunction

# Provide a few aliases
Context = ScallopContext
Provenance = ScallopProvenance
Generic = GenericTypeParameter
ff = foreign_function
fp = foreign_predicate
Map = InputMapping

# Loading
def __getattr__(name: str):
  forward_alias = ["ScallopForwardFunction", "ForwardFunction", "Module"]
  if name in forward_alias:
    from .forward import ScallopForwardFunction
    return ScallopForwardFunction
  elif name in globals():
    return globals()[name]
  else:
    raise AttributeError(f"Attribute `{name}` not found inside `scallopy`")
