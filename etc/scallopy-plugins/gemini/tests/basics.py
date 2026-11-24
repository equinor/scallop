import unittest
from unittest.mock import patch

from google.genai import types
import scallopy
import scallop_gemini

class TestBasics(unittest.TestCase):
  def test_ff_gemini(self):
    ctx = scallopy.Context()
    plugin = scallop_gemini.ScallopGeminiPlugin()

    plugin_registry = scallopy.PluginRegistry()
    plugin_registry.load_plugin(plugin)
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
         patch("google.genai.Client") as create_client:
      plugin_registry.configure({"gemini_model": "gemini-test", "gemini_temperature": 0.25})
    client = create_client.return_value
    client.models.generate_content.return_value = types.GenerateContentResponse(
      candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="test answer")]))],
    )
    plugin_registry.load_into_ctx(ctx)

    ctx.add_program("""
      rel questions = {(1, "test question")}
      rel answer(id, $gemini(x)) = questions(id, x)
      query answer
    """)
    ctx.run()

    self.assertEqual(list(ctx.relation("answer")), [(1, "test answer")])
    create_client.assert_called_once_with(api_key="test-key")
    client.models.generate_content.assert_called_once()
    request = client.models.generate_content.call_args.kwargs
    self.assertEqual(request["model"], "gemini-test")
    self.assertEqual(request["config"].temperature, 0.25)

  @unittest.skipUnless(scallopy.torch_tensor_enabled(), "requires Scallopy tensor support")
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

    self.assertEqual(len(list(ctx.relation("vectors"))), 2)
    embed.assert_called_once()
    request = embed.call_args.kwargs
    self.assertEqual(request["model"], "models/gemini-embedding-001")
    self.assertEqual(request["contents"], "example")
    self.assertEqual(request["config"].task_type, "RETRIEVAL_DOCUMENT")
    self.assertEqual(request["config"].output_dimensionality, 768)
