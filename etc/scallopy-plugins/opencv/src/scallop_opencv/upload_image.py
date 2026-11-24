import os

import requests
from torch import Tensor

from .save_image import _save_image

def _upload_imgur(
    img_tensor: Tensor,
    save_image_path: str = "",
) -> str:
  img_path = _save_image(img_tensor, save_image_path=save_image_path)

  if len(img_path) == 0:
    return ""

  client_id = os.environ.get("IMGUR_API_ID")
  if not client_id:
    return ""

  with open(img_path, "rb") as image:
    response = requests.post(
      "https://api.imgur.com/3/image",
      headers={"Authorization": f"Client-ID {client_id}"},
      files={"image": image},
      timeout=30,
    )

  response.raise_for_status()
  payload = response.json()
  if not payload.get("success"):
    raise RuntimeError("Imgur image upload failed")
  return payload["data"]["link"]
