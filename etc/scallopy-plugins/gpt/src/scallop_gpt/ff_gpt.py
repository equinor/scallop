from functools import cache

import scallopy

from .plugin import ScallopGPTPlugin, message_text

def get_gpt(plugin: ScallopGPTPlugin):
  @scallopy.foreign_function
  @cache
  def gpt(prompt: str) -> str:
    plugin.assert_can_request()
    plugin.increment_num_performed_request()
    response = plugin.client().chat.completions.create(
      model=plugin.model(),
      messages=[{"role": "user", "content": prompt}],
      temperature=plugin.temperature(),
    )
    return message_text(response.choices[0].message).strip()

  return gpt
