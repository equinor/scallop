import scallopy
from torch import Tensor

from ._mask import refine_mask

"""Code adapted from https://github.com/allenai/visprog.git"""
@scallopy.foreign_function
def color_pop(img_tensor: Tensor, mask_tensor: Tensor) -> Tensor:
  from PIL import Image
  import torch
  import numpy as np

  image = Image.fromarray(img_tensor.numpy())
  pixels = np.asarray(image)
  grayscale = np.asarray(image.convert("L"))[..., None]
  alpha = refine_mask(pixels.astype(np.uint8), mask_tensor).astype(float)[..., None]
  result = alpha * pixels + (1 - alpha) * grayscale
  return torch.tensor(result.astype(np.uint8))
