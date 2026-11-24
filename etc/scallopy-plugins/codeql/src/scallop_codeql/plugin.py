import os
from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Any
import yaml
import csv
import shutil
import subprocess

import scallopy

from .utils.constants import CUSTOM_CODEQL_QUERY_DIR

SUPPORTED_LANGUAGES = {
  "java": {
    "custom_script_path": "qlpacks/codeql/java-queries/0.8.3/scallop_codeql",
  }
}

class ScallopCodeQLPlugin(scallopy.Plugin):
  def __init__(self):
    super().__init__()

    # Configurations
    self._configured = False
    self._codeql_path = None
    self._codeql_exec = None

    # Project specific
    self._codeql_db = None
    self._codeql_db_yaml_dir = None
    self._codeql_db_yaml = None
    self._project_language: str | None = None

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--codeql-db", type=str, help="The CodeQL database of the target repository that we are going to analyze")
    parser.add_argument("--codeql-path", type=str, help="The directory to the `codeql` folder. The `codeql` executable should be within the folder")

  def configure(self, args: Dict = {}, unknown_args: List = []):
    # First check if the user has specified a codeql DB or not
    self._configured = False
    self._project_language = None
    self._codeql_db = args.get("codeql_db")
    if self._codeql_db is None:
      # Ignore
      return
    else:
      # Check that the database is well formed
      self._codeql_db_yaml_dir = f"{self._codeql_db}/codeql-database.yml"
      if not os.path.exists(self._codeql_db_yaml_dir):
        raise Exception("Incomplete CodeQL database; expected codeql-database.yml within the db directory")

      # Load the yaml file
      self._codeql_db_yaml = yaml.safe_load(open(self._codeql_db_yaml_dir))
      if not self._codeql_db_yaml["finalised"]:
        raise Exception("Incomplete CodeQL database; it is not finalised")

      # Check the language of the project
      self._project_language = self._codeql_db_yaml["primaryLanguage"]
      if self._project_language not in SUPPORTED_LANGUAGES:
        raise Exception(f"Unsupported project language `{self._project_language}`")

      # Get the source code directory of the project
      self._project_source_code_dir = self._codeql_db_yaml["sourceLocationPrefix"]

      # Create a temporary directory for Scallop-CodeQL
      self._cache_dir = os.path.join(self._codeql_db, "scallop_codeql", self._project_language)
      os.makedirs(self._cache_dir, exist_ok=True)

    # If there is a CodeQL DB from the arguments, we need to
    env_codeql_path = os.getenv("CODEQL_PATH")
    self._codeql_path = env_codeql_path if env_codeql_path is not None else args.get("codeql_path")
    if self._codeql_path is None:
      raise Exception("Missing CodeQL path; either set the environment variable CODEQL_PATH or pass as a command line argument `--codeql-path`")

    # Check if the codeql executable is there
    self._codeql_exec = f"{self._codeql_path}/codeql"
    if not os.path.exists(self._codeql_exec):
      raise Exception(f"[scallop_codeql] `codeql` executable not found under CODEQL_PATH: {self._codeql_path}")

    # At the end, the codeql plugin has been successfully initialized
    self._configured = True

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    # Register each predicate one-by-one
    from .attributes.codeql_database import get_codeql_database
    ctx.register_foreign_attribute(get_codeql_database(self))

    # ctx.register_foreign_attribute(get_codeql_source(self))

  def project_language(self) -> str | None:
    return self._project_language

  def is_java(self) -> bool:
    return self._project_language == "java"

  def run_codeql_query(self, query: str, output: str, overwrite: bool = False, debug: bool = False) -> str:
    """
    Run the given CodeQL query

    :param query, a relative directory from `CUSTOM_CODEQL_QUERY_DIR`
    :param output, the output file name (without bprs or csv)
    """
    language = self._project_language
    executable = self._codeql_exec
    if not self._configured or language is None or executable is None:
      raise RuntimeError("[scallop_codeql] Configure a CodeQL database and executable before running queries")

    # Directories
    query_file_dir = f"{CUSTOM_CODEQL_QUERY_DIR}/{query}"
    target_file_dir = f"{self._codeql_path}/{SUPPORTED_LANGUAGES[language]['custom_script_path']}/{query}"
    result_bqrs_file_dir = f"{self._cache_dir}/{output}.bqrs"
    result_csv_file_dir = f"{self._cache_dir}/{output}.csv"

    # Found in cache
    if os.path.exists(result_csv_file_dir) and not overwrite:
      if debug: print(f"[scallop_codeql] result CSV `{result_csv_file_dir}` found; returning directly")
      return result_csv_file_dir

    # 1. Copy the query to CodeQL directory
    Path(target_file_dir).parent.mkdir(parents=True, exist_ok=True)
    if debug: print(f"[scallop_codeql] copying query file to CodeQL directory `{target_file_dir}`")
    shutil.copyfile(query_file_dir, target_file_dir)

    # 2. Run the query and output to bprs
    codeql_run_query_cmd = [executable, "query", "run", f"--database={self._codeql_db}", f"--output={result_bqrs_file_dir}", "--", target_file_dir]
    if debug: print("[scallop_codeql] running CodeQL query...")
    run_query_output = subprocess.run(codeql_run_query_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert os.path.exists(result_bqrs_file_dir), f"[scallop_codeql] CodeQL Error: CodeQL failed to produce analysis result bqrs\n\nCodeQL STDOUT:\n{run_query_output.stderr}"
    if debug: print(f"[scallop_codeql] success; obtained bqrs file `{result_bqrs_file_dir}`")

    # 3. Convert the output to csv
    codeql_decode_bqrs_cmd = [executable, "bqrs", "decode", result_bqrs_file_dir, "--format=csv", f"--output={result_csv_file_dir}"]
    if debug: print("[scallop_codeql] running CodeQL decode...")
    decode_output = subprocess.run(codeql_decode_bqrs_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert os.path.exists(result_csv_file_dir), f"[scallop_codeql] CodeQL Error: CodeQL failed to decode bqrs into csv\n\nCodeQL STDOUT:\n{decode_output.stderr}"
    if debug: print(f"[scallop_codeql] success; obtained csv file `{result_csv_file_dir}`")

    # 4. Return the csv
    return result_csv_file_dir

  def try_load_index_csv(self, index_csv_name: str) -> str | None:
    csv_file_dir = f"{self._cache_dir}/{index_csv_name}.csv"
    if os.path.exists(csv_file_dir):
      return csv_file_dir
    else:
      return None

  def save_index_csv(self, index_csv_name: str, fields: List[str], data: Iterable[Iterable[Any]]) -> str:
    csv_file_dir = f"{self._cache_dir}/{index_csv_name}.csv"
    csv_writer = csv.writer(open(csv_file_dir, "w"))
    csv_writer.writerow(fields)
    csv_writer.writerows(data)
    return csv_file_dir
