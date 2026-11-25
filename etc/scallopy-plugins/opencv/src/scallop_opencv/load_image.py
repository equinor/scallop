import scallopy
from torch import Tensor


@scallopy.foreign_function
def load_image(image_dir: str) -> Tensor:
    # Import
    import torch
    import numpy
    from PIL import Image

    # Load image
    image = Image.open(image_dir).convert("RGB")
    image_tensor = torch.tensor(numpy.asarray(image))

    # Return
    return image_tensor


@scallopy.foreign_function
def load_image_url(image_url: str) -> Tensor:
    import torch
    import numpy
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        return torch.tensor(numpy.asarray(image.convert("RGB")))
