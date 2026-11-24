from google.genai import types
import scallopy

from . import ScallopGeminiPlugin
from ._response import candidate_text

def get_gemini(plugin: ScallopGeminiPlugin):
  # A storage for @gemini foreign predicate
  STORAGE = {}

  @scallopy.foreign_predicate
  def gemini(s: str) -> scallopy.Facts[None, tuple[str]]:
    # Check if the storage already contains the response
    if s in STORAGE:
      response = STORAGE[s]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Memoize the response
      plugin.increment_num_performed_request()

      response = plugin.client().models.generate_content(
        model=plugin.model(),
        contents=s,
        config=types.GenerateContentConfig(
          temperature=plugin.temperature(),
          top_k = 1,
          top_p = 0.5,
        ),
      )
      STORAGE[s] = response

    # Iterate through all the choices
    if not response.candidates:
      raise ValueError("[scallop_gemini] Response contains no candidates")
    for candidate in response.candidates:
      result = candidate_text(candidate).strip()
      yield (result,)

  return gemini
