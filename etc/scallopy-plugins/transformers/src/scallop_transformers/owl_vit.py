from typing import List, Optional
from collections.abc import Iterator
from functools import lru_cache
import logging
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import scallopy
from scallopy.utils.image import box_slices, save_debug_image
from scallop_gpu import get_device

FA_NAME = "owl_vit"
ERR_HEAD = f"[@{FA_NAME}]"

DUMP_IMAGE_PATH = ".tmp/scallop-owl-vit/"
HARD_OBJ_LIMIT = 1000

FIELDS = {
  "bbox-x": {
    "description": "The x-coordinate of the bounding box",
    "type": "u32",
  },
  "bbox-y": {
    "description": "The y-coordinate of the bounding box",
    "type": "u32",
  },
  "bbox-w": {
    "description": "The width of the bounding box",
    "type": "u32",
  },
  "bbox-h": {
    "description": "The height of the bounding box",
    "type": "u32",
  },
  "bbox-center-x": {
    "description": "The x coordinate of the center of the bounding box",
    "type": "u32",
  },
  "bbox-center-y": {
    "description": "The y coordinate of the center of the bounding box",
    "type": "u32",
  },
  "cropped-image": {
    "description": "The cropped image (bounded by bounding box)",
    "type": "Tensor",
  },
  "class": {
    "description": "The classified label",
    "type": "String",
  },
  "area": {
    "description": "The area of the bounding box",
    "type": "u32",
  },
}

@lru_cache(maxsize=1)
def get_owl_vit_model(checkpoint: str, device: str) -> tuple[OwlViTForObjectDetection, OwlViTProcessor]:
  processor = OwlViTProcessor.from_pretrained(checkpoint)
  model = OwlViTForObjectDetection.from_pretrained(checkpoint).to(device)
  return model, processor


@scallopy.foreign_attribute
def owl_vit(
  item,
  *,
  object_queries: Optional[List[str]] = None,
  output_fields: List[str] = ["class"],
  score_threshold: float = 0.1,
  score_multiplier: float = 1.0,
  input_obj_count: bool = False,
  expand_crop_region: int = 0,
  limit: Optional[int] = None,
  checkpoint: str = "google/owlvit-base-patch32",
  debug: bool = False,
  dump_image: bool = False,
  flatten_probability = False,
):
  # Needs to be a relation declaration, and can have only one relation
  assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations]"

  # Get the argument types
  relation_decl = item.relation_decl(0)
  arg_names = [ab.name.name for ab in relation_decl.arg_bindings]
  arg_types = [ab.ty for ab in relation_decl.arg_bindings]

  # Expected input arguments
  expected_args = [("Tensor", "image: Tensor")]

  # Object queries
  if object_queries is None:
    expected_args += [("String", "object_queries: String")]

  # Object count
  if input_obj_count:
    expected_args += [("u32", "obj_count: u32")]

  num_bounded = len(expected_args)

  # Expected output arguments
  expected_args += [("u32", "id: u32")]

  # Fields: make sure that all output fields are supported
  assert type(output_fields) == list, f"{ERR_HEAD} The parameter `output_fields` need to be a list"
  assert len(output_fields) > 0, f"{ERR_HEAD} The `output_fields` list must be non-empty"
  for field in output_fields:
    assert field in FIELDS, f"{ERR_HEAD} Unknown output field `{field}`; supported fields are {list(FIELDS.keys())}"
  expected_args += [(FIELDS[f]["type"], f + ": " + FIELDS[f]["type"]) for f in output_fields]

  # Check arity of relation
  assert len(arg_types) == len(expected_args), f"{ERR_HEAD} relation has to be of arity-{len(expected_args)}: {[s for (_, s) in expected_args]}"

  # Making sure that arg types are as expected
  for arg_id in range(len(arg_types)):
    (expected_ty, s), arg_name, arg_ty = expected_args[arg_id], arg_names[arg_id], arg_types[arg_id].name()
    if arg_ty.lower() != expected_ty.lower():
      if arg_name is not None:
        raise Exception(f"{ERR_HEAD} Type mismatch for `{arg_name}`. Expected `{s}`, but found arg of type `{arg_ty}`")
      else:
        raise Exception(f"{ERR_HEAD} Type mismatch for argument `{arg_id+1}`. Expected `{s}`, but found arg of type `{arg_ty}`")

  # Make sure object_queries is not empty if passed to attribute
  if object_queries:
    assert type(object_queries) == list, f"{ERR_HEAD} The parameter `object_queries` must be a list of strings"
    assert len(object_queries) > 0, f"{ERR_HEAD} The `object_queries` list must be non-empty"
    assert all([type(q) == str for q in object_queries]), f"{ERR_HEAD} The parameter `object_queries` must be a list of strings"

  # Generate foreign predicate
  @scallopy.foreign_predicate(
    name=relation_decl.name.name,
    input_arg_types=arg_types[:num_bounded],
    output_arg_types=arg_types[num_bounded:],
    tag_type=float,
  )
  def owl_vit_search(*args) -> Iterator[tuple[float, tuple[int | str | torch.Tensor, ...]]]:
    # If object_queries is an arg, assume it's a semicolon-separated string
    if object_queries is None and input_obj_count:
      assert len(args) == 3
      img = args[0]
      input_object_queries = args[1].split(";")
      obj_count_limit = min(args[2], limit) if limit else args[2]
    elif object_queries is None:
      assert len(args) == 2
      img = args[0]
      input_object_queries = args[1].split(";")
      obj_count_limit = limit if limit else HARD_OBJ_LIMIT
    elif input_obj_count:
      assert len(args) == 2
      img = args[0]
      input_object_queries = object_queries
      obj_count_limit = min(args[1], limit) if limit else args[1]
    else:
      assert len(args) == 1
      img = args[0]
      input_object_queries = object_queries
      obj_count_limit = limit if limit else HARD_OBJ_LIMIT

    device = get_device()
    try:
      owl_vit_model, owl_vit_processor = get_owl_vit_model(checkpoint, device)
    except (OSError, RuntimeError, ValueError):
      logging.getLogger(__name__).exception("Failed to load OWL-ViT checkpoint %s on %s", checkpoint, device)
      return

    img_h, img_w, _ = img.shape
    image = Image.fromarray(img.numpy())
    inputs = owl_vit_processor(text=input_object_queries, images=image, return_tensors="pt").to(device)

    owl_vit_model.eval()
    with torch.no_grad():
      outputs = owl_vit_model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)

    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = owl_vit_processor.image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=score_threshold,
        target_sizes=target_sizes,
    )

    # Sort results by score descending
    zipped = list(zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]))
    zipped.sort(key=lambda x: x[0], reverse=True)

    # Keep track of count to stay within limit
    yielded_count = 0

    for (segment_id, (score, label, box)) in enumerate(zipped):
      if yielded_count >= obj_count_limit: break
      yielded_count += 1

      # Basic data
      final_score = float(score) * score_multiplier if not flatten_probability else 1.0
      bbox_raw = [max(round(coord), 0) for coord in box.tolist()]
      bbox_x, bbox_y = bbox_raw[0], bbox_raw[1]
      bbox_w, bbox_h = bbox_raw[2] - bbox_x, bbox_raw[3] - bbox_y

      # Output tuple
      result_tuple: list[int | str | torch.Tensor] = [segment_id]
      for field in output_fields:
        if field == "bbox-x":
          result_tuple.append(bbox_x)
        elif field == "bbox-y":
          result_tuple.append(bbox_y)
        elif field == "area":
          result_tuple.append(bbox_w * bbox_h)
        elif field == "bbox-w":
          result_tuple.append(bbox_w)
        elif field == "bbox-h":
          result_tuple.append(bbox_h)
        elif field == "bbox-center-x":
          result_tuple.append(int(bbox_x + bbox_w / 2))
        elif field == "bbox-center-y":
          result_tuple.append(int(bbox_y + bbox_h / 2))
        elif field == "class":
          result_tuple.append(input_object_queries[label])
        elif field == "cropped-image":
          rows, columns = box_slices(img_h, img_w, bbox_x, bbox_y, bbox_w, bbox_h, expand_crop_region)
          cropped_image = img[rows, columns, :]
          result_tuple.append(cropped_image)
          if dump_image:
            save_debug_image(DUMP_IMAGE_PATH, id(img), segment_id, field, cropped_image)

      if debug:
        print((final_score, tuple(result_tuple)))

      yield (final_score, tuple(result_tuple))

  return owl_vit_search
