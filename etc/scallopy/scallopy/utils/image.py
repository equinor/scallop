from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..torch_importer import Tensor


def box_slices(
  height: int,
  width: int,
  x: int,
  y: int,
  box_width: int,
  box_height: int,
  padding: int = 0,
) -> tuple[slice, slice]:
  """Return row and column slices for a padded box, clipped to the image."""
  rows = slice(max(0, y - padding), min(height, y + box_height + padding))
  columns = slice(max(0, x - padding), min(width, x + box_width + padding))
  return rows, columns


def save_debug_image(root: str, image_id: int, index: int, kind: str, tensor: Tensor) -> None:
  from PIL import Image

  directory = Path(root) / str(image_id)
  directory.mkdir(parents=True, exist_ok=True)
  Image.fromarray(tensor.numpy()).save(directory / f"{kind}-{index}.jpg")
