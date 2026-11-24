import scallopy

from .plugin import ScallopGPTPlugin, message_text

def get_gpt(plugin: ScallopGPTPlugin):
  # A storage for @gpt foreign predicate
  STORAGE = {}

  @scallopy.foreign_predicate
  def gpt(s: str) -> scallopy.Facts[None, tuple[str]]:
    # Check if the storage already contains the response
    if s in STORAGE:
      response = STORAGE[s]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Memoize the response
      plugin.increment_num_performed_request()
      response = plugin.client().chat.completions.create(
        model=plugin.model(),
        messages=[{"role": "user", "content": s}],
        temperature=plugin.temperature(),
      )
      STORAGE[s] = response

    # Iterate through all the choices
    for choice in response.choices:
      result = message_text(choice.message).strip()
      yield (result,)

  return gpt
