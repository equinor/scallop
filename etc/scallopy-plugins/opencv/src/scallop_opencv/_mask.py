import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def refine_mask(image: NDArray[np.uint8], mask: Tensor) -> NDArray[np.uint8]:
  import cv2

  refined = np.ascontiguousarray(mask).astype(np.uint8)
  cv2.grabCut(
    image,
    refined,
    (0, 0, image.shape[1], image.shape[0]),  # Ignored in GC_INIT_WITH_MASK mode.
    np.zeros((1, 65), np.float64),
    np.zeros((1, 65), np.float64),
    5,
    cv2.GC_INIT_WITH_MASK,
  )
  return refined
