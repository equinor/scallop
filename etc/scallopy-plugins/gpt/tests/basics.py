import unittest
from unittest.mock import patch

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion
import scallopy
import scallop_gpt


def completion(content):
  return ChatCompletion.model_validate({
    "id": "test-completion",
    "model": "test-model",
    "created": 0,
    "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
  })


class TestBasics(unittest.TestCase):
  def setUp(self):
    self.ctx = scallopy.Context()
    plugin = scallop_gpt.ScallopGPTPlugin()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("scallop_gpt.plugin.OpenAI") as create_client:
      plugin.configure({"openai_gpt_model": "test-model", "openai_gpt_temperature": 0.25})
    self.client = create_client.return_value
    self.complete = self.client.chat.completions.create
    plugin.load_into_ctx(self.ctx)

  def test_ff_gpt(self):
    self.complete.return_value = completion(" test answer ")
    self.ctx.add_program("""
      rel questions = {(1, "test question")}
      rel answer(id, $gpt(x)) = questions(id, x)
      query answer
    """)
    self.ctx.run()

    self.assertEqual(list(self.ctx.relation("answer")), [(1, "test answer")])
    self.complete.assert_called_once_with(
      model="test-model",
      messages=[{"role": "user", "content": "test question"}],
      temperature=0.25,
    )

  def test_fp_gpt(self):
    self.complete.return_value = completion(" test answer ")
    self.ctx.add_program("""
      rel questions = {("test question")}
      rel answers(a) = questions(q), gpt(q, a)
      query answers
    """)
    self.ctx.run()

    self.assertEqual(list(self.ctx.relation("answers")), [("test answer",)])

  def test_gpt_attribute(self):
    self.complete.return_value = completion('{"answer": "test answer"}')
    self.ctx.add_program("""
      @gpt(prompt="Answer {{question}}: {{answer}}")
      type complete(bound question: String, answer: String)
      rel questions = {("test question")}
      rel answers(a) = questions(q), complete(q, a)
      query answers
    """)
    self.ctx.run()

    self.assertEqual(list(self.ctx.relation("answers")), [("test answer",)])

  def test_gpt_extract_info(self):
    self.complete.side_effect = [completion("test reasoning"), completion('{"answer": "test answer"}')]
    self.ctx.add_program("""
      @gpt_extract_info(prompts=["Extract the answer"], cot=[true])
      type extract(bound question: String, answer: String)
      rel questions = {("test question")}
      rel answers(a) = questions(q), extract(q, a)
      query answers
    """)
    self.ctx.run()

    self.assertEqual(list(self.ctx.relation("answers")), [("test answer",)])
    self.assertEqual(self.complete.call_count, 2)
    self.assertEqual(self.complete.call_args.kwargs["response_format"], {"type": "json_object"})

  @unittest.skipUnless(scallopy.torch_tensor_enabled() and scallopy.torch_importer.has_pytorch(), "requires Scallopy tensor support and PyTorch")
  def test_encoder(self):
    embed = self.client.embeddings.create
    embed.return_value = CreateEmbeddingResponse.model_validate({
      "model": "test-embedding-model",
      "object": "list",
      "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
      "usage": {"prompt_tokens": 1, "total_tokens": 1},
    })
    self.ctx.add_program("""
      @gpt_encoder(model="test-embedding-model")
      type $embed(String) -> Tensor
      rel documents = {(1, "example"), (2, "example")}
      rel vectors(id, $embed(text)) = documents(id, text)
      query vectors
    """)
    self.ctx.run()

    self.assertEqual(len(list(self.ctx.relation("vectors"))), 2)
    embed.assert_called_once_with(input=["example"], model="test-embedding-model")
