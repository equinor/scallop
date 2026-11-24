import scallopy
from torch import Tensor

from ._mask import refine_mask

"""Code adapted from https://github.com/allenai/visprog.git"""

@scallopy.foreign_function
def bg_blur(img_tensor: Tensor, mask_tensor: Tensor) -> Tensor:
  from PIL import Image, ImageFilter
  import torch
  import numpy as np

  image = Image.fromarray(img_tensor.numpy())
  pixels = np.asarray(image)
  blur = ImageFilter.GaussianBlur(radius=5)
  blurred = np.asarray(image.filter(blur))

  mask = refine_mask(pixels.astype(np.uint8), mask_tensor)
  soft_mask = Image.fromarray(255 * mask).filter(blur)
  alpha = np.asarray(soft_mask)[..., None] / 255
  result = alpha * pixels + (1 - alpha) * blurred
  return torch.tensor(result.astype(np.uint8))
