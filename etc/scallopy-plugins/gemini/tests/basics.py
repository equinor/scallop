import unittest
from unittest.mock import patch

from google.genai import types
import scallopy
import scallop_gemini
from scallop_gemini._response import response_text


def completion(text):
  return types.GenerateContentResponse(
    candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))],
  )


class TestBasics(unittest.TestCase):
  def test_empty_responses_report_what_is_missing(self):
    for response, error in (
      (types.GenerateContentResponse(), "no candidates"),
      (types.GenerateContentResponse(candidates=[types.Candidate()]), "no text"),
    ):
      with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
        response_text(response)

  def test_ff_gemini(self):
    ctx = scallopy.Context()
    plugin = scallop_gemini.ScallopGeminiPlugin()

    plugin_registry = scallopy.PluginRegistry()
    plugin_registry.load_plugin(plugin)
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
         patch("google.genai.Client") as create_client:
      plugin_registry.configure({"gemini_model": "gemini-test", "gemini_temperature": 0.25})
    client = create_client.return_value
    client.models.generate_content.return_value = completion("test answer")
    plugin_registry.load_into_ctx(ctx)

    ctx.add_program("""
      rel questions = {(1, "test question"), (2, "test question")}
      rel answer(id, $gemini(x)) = questions(id, x)
      query answer
    """)
    ctx.run()

    self.assertCountEqual(ctx.relation_tuples("answer"), [(1, "test answer"), (2, "test answer")])
    create_client.assert_called_once_with(api_key="test-key")
    client.models.generate_content.assert_called_once()
    request = client.models.generate_content.call_args.kwargs
    self.assertEqual(request["model"], "gemini-test")
    self.assertEqual(request["config"].temperature, 0.25)

  def test_extract_info_only_structures_the_final_response(self):
    ctx = scallopy.Context()
    plugin = scallop_gemini.ScallopGeminiPlugin()
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
         patch("google.genai.Client") as create_client:
      plugin.configure()
    generate = create_client.return_value.models.generate_content
    generate.side_effect = [
      completion("test reasoning"),
      completion('{"rows": [{"answer": "test answer", "count": 2}]}'),
    ]
    plugin.load_into_ctx(ctx)

    ctx.add_program("""
      @gemini_extract_info(prompts=["Extract the answer and count"], cot=[true])
      type extract(bound question: String, answer: String, count: i32)
      rel questions = {(1, "test question"), (2, "test question")}
      rel answers(id, a, n) = questions(id, q), extract(q, a, n)
      query answers
    """)
    ctx.run()

    self.assertCountEqual(ctx.relation_tuples("answers"), [(1, "test answer", 2), (2, "test answer", 2)])
    reasoning, final = [request.kwargs["config"] for request in generate.call_args_list]
    self.assertIsNone(reasoning.response_json_schema)
    self.assertEqual(final.response_mime_type, "application/json")
    fields = final.response_json_schema["properties"]["rows"]["items"]
    self.assertEqual(fields["properties"], {"answer": {"type": "string"}, "count": {"type": "integer"}})
    self.assertEqual(fields["required"], ["answer", "count"])

  @unittest.skipUnless(scallopy.torch_tensor_enabled() and scallopy.torch_importer.has_pytorch(), "requires Scallopy tensor support and PyTorch")
  def test_encoder_uses_configured_client_and_attribute_model(self):
    ctx = scallopy.Context()
    plugin = scallop_gemini.ScallopGeminiPlugin()
    plugin_registry = scallopy.PluginRegistry()
    plugin_registry.load_plugin(plugin)
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
         patch("google.genai.Client") as create_client:
      plugin_registry.configure()
    embed = create_client.return_value.models.embed_content
    embed.return_value = types.EmbedContentResponse(
      embeddings=[types.ContentEmbedding(values=[0.25] * 768)],
    )
    plugin_registry.load_into_ctx(ctx)

    ctx.add_program("""
      @gemini_encoder(model="models/gemini-embedding-001")
      type $embed(String) -> Tensor
      rel documents = {(1, "example"), (2, "example")}
      rel vectors(id, $embed(text)) = documents(id, text)
      query vectors
    """)
    ctx.run()

    self.assertEqual(len(list(ctx.relation_tuples("vectors"))), 2)
    embed.assert_called_once()
    request = embed.call_args.kwargs
    self.assertEqual(request["model"], "models/gemini-embedding-001")
    self.assertEqual(request["contents"], "example")
    self.assertEqual(request["config"].task_type, "RETRIEVAL_DOCUMENT")
    self.assertEqual(request["config"].output_dimensionality, 768)
