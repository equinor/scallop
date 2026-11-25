from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
  from ..syntax import AstTypeNode


def relation_rows_schema(names: Sequence[str], types: Sequence[AstTypeNode]) -> dict[str, object]:
  json_types = {str: "string", bool: "boolean", int: "integer", float: "number"}
  row = {
    "type": "object",
    "properties": {name: {"type": json_types[ty.to_python_type()]} for name, ty in zip(names, types, strict=True)},
    "required": list(names),
    "additionalProperties": False,
  }
  return {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": row}},
    "required": ["rows"],
    "additionalProperties": False,
  }
