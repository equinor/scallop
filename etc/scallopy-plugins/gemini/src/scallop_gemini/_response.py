from google.genai import types


def candidate_text(candidate: types.Candidate) -> str:
  content = candidate.content
  if content is not None and content.parts:
    text = content.parts[0].text
    if text is not None:
      return text
  raise ValueError("[scallop_gemini] Response candidate contains no text")


def response_text(response: types.GenerateContentResponse) -> str:
  if not response.candidates:
    raise ValueError("[scallop_gemini] Response contains no candidates")
  return candidate_text(response.candidates[0])
