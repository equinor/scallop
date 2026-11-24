from google.genai import types

import scallopy

from . import ScallopGeminiPlugin

def get_gemini(plugin: ScallopGeminiPlugin):
  # For memoization
  STORAGE = {}

  @scallopy.foreign_function
  def gemini(prompt: str) -> str:
    if prompt in STORAGE:
      return STORAGE[prompt]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Add performed requests
      plugin.increment_num_performed_request()

      response = plugin.client().models.generate_content(
          model=plugin.model(),
          contents=prompt,
          config=types.GenerateContentConfig(
              temperature=plugin.temperature(),
              top_k = 1,
              top_p = 0.5,
          ),
        )

      result = response.candidates[0].content.parts[0].text
      # print("result :", result, "\n\n\n")
      # Store in the storage
      STORAGE[prompt] = result
      return result

  return gemini