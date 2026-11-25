from functools import cache

from google.genai import types

import scallopy

from . import ScallopGeminiPlugin
from ._response import response_text

def get_gemini(plugin: ScallopGeminiPlugin):
  @scallopy.foreign_function
  @cache
  def gemini(prompt: str) -> str:
    plugin.assert_can_request()
    plugin.increment_num_performed_request()
    response = plugin.client().models.generate_content(
      model=plugin.model(),
      contents=prompt,
      config=types.GenerateContentConfig(
        temperature=plugin.temperature(),
        top_k=1,
        top_p=0.5,
      ),
    )
    return response_text(response)

  return gemini