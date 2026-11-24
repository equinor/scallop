import scallopy

from .plugin import ScallopGPTPlugin, message_text

def get_gpt(plugin: ScallopGPTPlugin):
  # For memoization
  STORAGE = {}

  @scallopy.foreign_function
  def gpt(prompt: str) -> str:
    if prompt in STORAGE:
      return STORAGE[prompt]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Add performed requests
      plugin.increment_num_performed_request()
      response = plugin.client().chat.completions.create(
        model=plugin.model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=plugin.temperature(),
      )
      result = message_text(response.choices[0].message).strip()

      # Store in the storage
      STORAGE[prompt] = result
      return result

  return gpt
