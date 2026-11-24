import logging
import os

from torch import Tensor

def _save_image(
    img_tensor: Tensor,
    img_name: str | None = None,
    save_image_path: str = "",
) -> str:
  from PIL import Image

  # Convert image
  img = Image.fromarray(img_tensor.numpy())

  # Default name
  if img_name is None:
    img_name = str(id(img))

  # First get the directory
  directory = save_image_path
  if not os.path.exists(directory):
    os.makedirs(directory)
  file_name = os.path.join(directory, f"{img_name}.jpg")

  # Save the image
  try:
    img.save(file_name)
    return file_name
  except (OSError, ValueError):
    logging.getLogger(__name__).exception("Failed to save image to %s", file_name)
    return ""
