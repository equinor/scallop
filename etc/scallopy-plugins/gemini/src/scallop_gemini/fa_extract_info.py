from functools import cache
from typing import Callable, List, Optional, Tuple
import re
import json
from google.genai import types
import scallopy
from scallopy.utils.relation_schema import relation_rows_schema

from . import ScallopGeminiPlugin
from ._response import response_text

FA_NAME = "gemini_extract_info"
ERR_HEAD = f"[@{FA_NAME}]"


def get_gemini_extract_info(plugin: ScallopGeminiPlugin):
  SYS_PROMPT = "You are a helpful assistant and diligent programmer. Please answer everything in JSON format."
  COT_PROMPT = "Please give your reasoning step by step before giving the output."

  @scallopy.foreign_attribute
  def gemini_extract_info(
    item,
    *,
    header: str = "",
    prompts: List[str],
    examples: List[Tuple[List[str], List[List[Tuple[str, ...]]]]] = [],
    model: Optional[str] = None,
    cot: Optional[List[bool]] = None,
    error_retry_limit: int = 1,
    error_retry_message: str = "Sorry, the answer is invalid. We expect you to generate a JSON list.",
    debug: bool = False,
  ):
    # Check if the annotation is on relation type decl
    assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
    relation_decl_infos = [get_rel_decl_info(r) for r in item.relation_decls()]
    
    # # Make sure that all the relation type declarations have the same bounded arguments (name and type)
    (_, fst_arg_names, fst_arg_types, _, fst_num_bounded) = relation_decl_infos[0]
    input_arg_names = fst_arg_names[:fst_num_bounded]
    input_arg_types = fst_arg_types[:fst_num_bounded]

    for (i, at) in enumerate(input_arg_types):
      assert at.is_string(), f"{ERR_HEAD} The `{i+1}`th input variable has to be string, found `{at}`"
    for (_, curr_arg_names, curr_arg_types, _, curr_num_bounded) in relation_decl_infos[1:]:
      assert curr_num_bounded == fst_num_bounded, f"{ERR_HEAD} The number of input variables are differing"
      for j in range(fst_num_bounded):
        an, at = curr_arg_names[j], curr_arg_types[j]
        assert an == input_arg_names[j], f"{ERR_HEAD} The `{j+1}`th input variables need to have the same name, found `{an}`"
        assert at.is_string(), f"{ERR_HEAD} The `{j+1}`th input variable has to be a string, found `{at}`"

    # # Make sure that every relation declaration gets a prompt
    assert len(prompts) == len(relation_decl_infos), f"{ERR_HEAD} The number of prompts must match the number of relations"
    
    # # Make sure examples are formatted correctly
    for (bound, output) in examples:
      assert len(bound) == fst_num_bounded, f"{ERR_HEAD} The number of input variables in each example must match the number of input variables in the relations"
      assert len(output) == len(relation_decl_infos), f"{ERR_HEAD} The number of output relations in each example must match the number of declared relations"
      for i in range(len(output)):
        for fact in output[i]:
          assert len(fact) == len(relation_decl_infos[i][1]) - fst_num_bounded, f"{ERR_HEAD} The arity of the fact tuples in each example must match the arity of their corresponding declared relation"

    # # Make sure that every relation declaration gets a cot
    assert cot is None or len(cot) == len(relation_decl_infos), f"{ERR_HEAD} The length of `cot` must match the number of relations"

    # # Get the gemini invoker
    gemini_invoker = generate_gemini_invoker(relation_decl_infos, input_arg_names, header, prompts, examples, model, cot, debug)

    # # Get the foreign predicates
    fps = [generate_foreign_predicate(gemini_invoker, relation_decl_infos[i]) for i in range(len(relation_decl_infos))]
    return fps

  def generate_foreign_predicate(
    gemini_invoker: Callable,
    relation_type
  ):
    # Get the relation name and argument types
    (name, arg_names, arg_types, _, num_bounded) = relation_type
    
    # Get the foreign predicate for invoking gemini
    @scallopy.foreign_predicate(
      name,
      input_arg_types=arg_types[:num_bounded],
      output_arg_types=arg_types[num_bounded:],
      tag_type=None)
    def invoke_gemini_and_get_result(*args):
      for row in gemini_invoker(*args)[name]:
        yield tuple(ty.parse_value(row[arg]) for arg, ty in zip(arg_names[num_bounded:], arg_types[num_bounded:]))

    return invoke_gemini_and_get_result


  def generate_gemini_invoker(
    relation_type_infos: List,
    input_arg_names: List[str],
    header: str,
    prompts: List[str],
    examples: List[Tuple[List[str], List[List[Tuple[str, ...]]]]],
    model: Optional[str] = None,
    cot: Optional[List[bool]] = None,
    debug: bool = False,
  ):
    def format_input(args):
      return ''.join([f"{arg_name}: {arg_value}\n" for (arg_name, arg_value) in zip(input_arg_names, args)])
    
    @cache
    def invoke_gemini(*args):
      # Get the model to use
      if model is None: local_model = plugin.model()
      else: local_model = model
      

      # Turn cot on or off
      if cot is None: cot_arr = [False] * len(relation_type_infos)
      else: cot_arr = cot

      # Check whether we can call gemini
      plugin.assert_can_request()
      plugin.increment_num_performed_request()
      client = plugin.client()

      # The result to return
      result = {}

      # Debug
      if debug: print(f"Processing question: {args}")

      # Current messages
      current_messages = SYS_PROMPT + header + "\n"

      # Add examples
      for (bound, output) in examples:
        for (idx, ((_, arg_names, arg_types, _, num_bounded), prompt, output_arg_values)) in enumerate(zip(relation_type_infos, prompts, output)):
          call_header = format_input(bound) if idx == 0 else ""
          current_messages += call_header + prompt + " "

          rows = [
            {name: ty.parse_value(value) for name, ty, value in zip(arg_names[num_bounded:], arg_types[num_bounded:], values)}
            for values in output_arg_values
          ]
          current_messages += json.dumps({"rows": rows}) + "\n"
        current_messages += "\n"

      # Make the subsequent calls to Gemini
      for (idx, (relation_type_info, prompt)) in enumerate(zip(relation_type_infos, prompts)):
        (name, arg_names, arg_types, _, num_bounded) = relation_type_info
        call_header = format_input(args) if idx == 0 else ""
        cot_response_content = ""
        if cot_arr[idx]:
          cot_message = call_header + prompt + COT_PROMPT
          cot_response = client.models.generate_content(
            model=local_model,
            contents=current_messages + cot_message,
            config=types.GenerateContentConfig(
              temperature=plugin.temperature(),
              top_k=1,
              top_p=0.5,
            ),
          )
          cot_response_content = response_text(cot_response)
        current_message = call_header + cot_response_content + prompt
        
        if debug: print(f"> Sending request: {current_message}")
        current_messages += current_message
        
        curr_response = client.models.generate_content(
          model=local_model,
          contents=current_messages,
          config=types.GenerateContentConfig(
            temperature=plugin.temperature(),
            top_k=1,
            top_p=0.5,
            response_mime_type="application/json",
            response_json_schema=relation_rows_schema(arg_names[num_bounded:], arg_types[num_bounded:]),
          ),
        )
        curr_response_message = response_text(curr_response)

        if debug: print(f"> Obtained response: {curr_response_message}\n")
        current_messages += curr_response_message
        result[name] = json.loads(curr_response_message)["rows"]
      return result

    return invoke_gemini

  
  def get_rel_decl_info(rel_decl):
    arg_names = [ab.name.name for ab in rel_decl.arg_bindings]
    arg_types = [ab.ty for ab in rel_decl.arg_bindings]
    pattern = "".join([get_boundness(ab.adornment) for ab in rel_decl.arg_bindings])

    # Get the pattern
    regex_match = re.match("^(b*)(f+)$", pattern)
    assert regex_match is not None, f"{ERR_HEAD} pattern must start with b (optional) and ending with f (required)"

    # Compute the number of `b`ounds and the number of `f`rees.
    num_bounded = len(regex_match[1])

    return (rel_decl.name.name, arg_names, arg_types, pattern, num_bounded)


  def get_boundness(adornment) -> str:
    return "b" if adornment and adornment.is_bound() else "f"


  return gemini_extract_info
