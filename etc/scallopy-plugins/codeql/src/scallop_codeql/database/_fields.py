from collections.abc import Callable
from typing import TypedDict


class FieldMetadata(TypedDict):
  name: str
  type: str


class CsvField(FieldMetadata):
  retrieve: Callable[[int, list[str]], int | str]
