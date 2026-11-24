from google.genai import types
import torch

import scallopy
from scallopy.plugin.utils import get_device

from . import ScallopGeminiPlugin

FA_NAME = "gemini_encoder"
ERR_HEAD = f"[@{FA_NAME}]"

def get_gemini_encoder(plugin: ScallopGeminiPlugin):

  @scallopy.foreign_attribute
  def gemini_encoder(item, *, debug: bool = False, model: str = "gemini-embedding-001"):
    # Check if the annotation is on function type decl
    assert item.is_function_decl(), f"{ERR_HEAD} has to be an attribute of a function type declaration"

    # Check the input and return types
    arg_types = item.function_decl_arg_types()
    assert len(arg_types) == 1 and arg_types[0].is_string(), f"{ERR_HEAD} expects only one `String` argument"
    assert item.function_decl_ret_type().is_tensor(), f"{ERR_HEAD} expects that the return type is `Tensor`"

    STORAGE = {}

    # Generate foreign function
    @scallopy.foreign_function(name=item.function_decl_name())
    def encode_text(text: str) -> torch.Tensor:
      # Check memoization
      if text not in STORAGE:
        # Make sure that we can do request
        plugin.assert_can_request()

        if debug:
          print(f"{ERR_HEAD} Querying `{model}` for text `{text}`")

        # Request the embedding
        response = plugin.client().models.embed_content(
          model=model,
          contents=text,
          config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
        )

        if not response.embeddings or not response.embeddings[0].values:
          raise RuntimeError(f"{ERR_HEAD} Gemini returned no embedding")
        embedding = response.embeddings[0].values

        if debug:
          print(f"{ERR_HEAD} Obtaining embedding: {embedding}")

        STORAGE[text] = embedding
        plugin.increment_num_performed_request()

      # Return tensor
      device = get_device()
      result_embedding = STORAGE[text]
      result = torch.tensor(result_embedding).to(device=device)
      return result

    return encode_text

  return gemini_encoder
