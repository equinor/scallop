from typing import Optional
from functools import lru_cache
import logging
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

import scallopy
from scallop_gpu import get_device


FA_NAME = "vilt"
ERR_HEAD = f"[@{FA_NAME}]"

HARD_TOP_LIMIT = 1000

@lru_cache(maxsize=1)
def get_vilt_model(checkpoint: str, device: str) -> tuple[ViltForQuestionAnswering, ViltProcessor]:
  processor = ViltProcessor.from_pretrained(checkpoint)
  model = ViltForQuestionAnswering.from_pretrained(checkpoint).to(device)
  return model, processor


@scallopy.foreign_attribute
def vilt(
  item,
  *,
  question: Optional[str] = None,
  top: int = 5,
  score_threshold: float = 0.1,
  checkpoint: str = "dandelin/vilt-b32-finetuned-vqa",
  debug: bool = False
):
  # Needs to be a relation declaration, and can have only one relation
  assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations]"

  # Get the relation name and argument types
  relation_decl = item.relation_decl(0)
  arg_types = [arg.ty for arg in relation_decl.arg_bindings]

  # Check the argument types
  if question is not None: assert len(arg_types) == 2, f"{ERR_HEAD} relation with no default question has to be of arity-2"
  else: assert len(arg_types) == 3, f"{ERR_HEAD} relation with default question has to be of arity-3"
  assert arg_types[0].is_tensor(), f"{ERR_HEAD} first argument has to be of type `Tensor`"
  assert arg_types[1].is_string(), f"{ERR_HEAD} second argument has to be of type `String`"
  if question is None: assert arg_types[2].is_string(), f"{ERR_HEAD} third argument has to be of type `String`"

  # Check if top is valid
  assert type(top) is int, f"{ERR_HEAD} `top` must be an integer, not {type(top)}"
  assert 1 <= top <= HARD_TOP_LIMIT, f"{ERR_HEAD} `top` must be within the interval [1, {HARD_TOP_LIMIT}]"

  # Generate the foreign predicate
  @scallopy.foreign_predicate(
      name=relation_decl.name.name,
      input_arg_types=arg_types[:-1],
      output_arg_types=arg_types[-1:],
      tag_type=float)
  def vilt_vqa(*args):
    if question is None:
      assert len(args) == 2
      input_question = args[1]
    else:
      assert len(args) == 1
      input_question = question

    img = args[0]
    device = get_device()
    try:
      vilt_model, vilt_processor = get_vilt_model(checkpoint, device)
    except (OSError, RuntimeError, ValueError):
      logging.getLogger(__name__).exception("Failed to load ViLT checkpoint %s on %s", checkpoint, device)
      return

    image = Image.fromarray(img.numpy())
    encoding = vilt_processor(image, input_question, return_tensors="pt").to(device)
    with torch.no_grad(): outputs = vilt_model(**encoding)

    probabilities = outputs.logits[0].sigmoid()
    scores, indices = probabilities.topk(min(top, probabilities.numel()))
    labels = vilt_model.config.id2label
    if labels is None:
      raise ValueError(f"{ERR_HEAD} checkpoint `{checkpoint}` does not define answer labels")

    for score, index in zip(scores.tolist(), indices.tolist()):
      answer = labels[index]

      if debug:
        print((score, (input_question, answer)))

      if score >= score_threshold:
        yield (score, (answer,))

  return vilt_vqa
