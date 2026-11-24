from functools import wraps
from typing import Callable, Concatenate, Mapping, ParamSpec, Protocol, TypeVar

class HistoryAction:
  def __init__(self, func_name: str, pos_args: tuple[object, ...], kw_args: Mapping[str, object]):
    self.func_name = func_name
    self.pos_args = pos_args
    self.kw_args = kw_args

  def __repr__(self):
    pos_args_str = ", ".join([repr(a) for a in self.pos_args])
    if len(self.kw_args) == 0:
      return f"{self.func_name}({pos_args_str})"
    else:
      kw_args_str = ", ".join([f"{k}={v}" for (k, v) in self.kw_args.items()])
      return f"{self.func_name}({pos_args_str}, {kw_args_str})"


class _HistoryRecorder(Protocol):
  _history_actions: list[HistoryAction]


_Context = TypeVar("_Context", bound=_HistoryRecorder)
_Args = ParamSpec("_Args")
_Result = TypeVar("_Result")


def record_history(f: Callable[Concatenate[_Context, _Args], _Result]) -> Callable[Concatenate[_Context, _Args], _Result]:
  func_name = getattr(f, "__name__")

  @wraps(f)
  def wrapper(this: _Context, /, *pos_args: _Args.args, **kw_args: _Args.kwargs) -> _Result:
    action = HistoryAction(func_name, pos_args, kw_args)
    this._history_actions.append(action)
    return f(this, *pos_args, **kw_args)
  return wrapper
