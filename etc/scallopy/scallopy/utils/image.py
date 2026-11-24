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
