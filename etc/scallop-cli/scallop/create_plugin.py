from typing import Optional
from pathlib import Path
import subprocess
import re
import argparse

PY_PROJECT_SRC = """\
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "scallop-{dashed_name}"
version = "{version}"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."scallop.plugin"]
{dashed_name} = "scallop_{underscore_name}:{plugin_name}"
"""


MAKEFILE_SRC = """\
.PHONY: build install develop clean

develop:
\tuv pip install --editable .

install: build
\tuv pip install --reinstall-package scallop-{dashed_name} dist/*.whl

build:
\tuv build --clear

clean:
\trm -rf dist
"""


GITIGNORE_SRC = """\
*.egg-info
.venv/
dist
"""


INIT_SRC = """\
import argparse
import scallopy

class {plugin_name}(scallopy.Plugin):
  def __init__(self):
    super().__init__()

  def setup_argparse(self, parser: argparse.ArgumentParser):
    pass

  def configure(self, args={{}}, unknown_args=[]):
    pass

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    pass
"""


DONE_PRINT = """\
Initialized empty Scallop plugin in {path}"""


ALLOWED_DIR_OBJS = set([
  ".DS_Store",
  "Makefile",
  "makefile",
  "readme.md",
  "Readme.md",
  "readme",
  "Readme",
  ".vscode",
  ".git",
  ".github",
])


def do_prompt_and_get_answer(
    prompt: str,
    require_non_empty: bool = True,
    default_value: Optional[str] = None,
    pattern: Optional[str | re.Pattern[str]] = None,
    empty_string_print: str = "[Error] The answer cannot be empty; please try again",
    pattern_fail_print: str = "[Error] Does not match the pattern {pattern}",
    failed_print: str = "[Error] Failed to get a proper answer; please try `create-scallopy-plugin` again",
    trim_whitespace: bool = True,
    num_try: Optional[int] = 3
) -> Optional[str]:
  try_count = 0
  while num_try is None or try_count < num_try:
    try_count += 1
    ans = input(prompt)
    if trim_whitespace:
      ans = ans.strip()
    if ans == "":
      if require_non_empty:
        print(empty_string_print)
      else:
        return default_value
    else:
      if pattern is None or re.search(pattern, ans) is not None:
        return ans
      else:
        print(pattern_fail_print.format(pattern=str(pattern)))
  print(failed_print)
  exit(1)


def is_git_repo(directory: Path) -> bool:
  result = subprocess.run(["git", "rev-parse"], cwd=directory, capture_output=True)
  return result.returncode == 0


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("create-plugin")
  parser.add_argument("path", nargs="?", default=None)
  args = parser.parse_args()

  # Check whether path is provided in the arguments
  cwd = Path(args.path) if args.path is not None else Path.cwd()
  cwd.mkdir(parents=True, exist_ok=True)

  # Check if the current folder does not contain `pyproject.toml`, `makefile`, and `src`
  curr_dir_objs = {entry.name for entry in cwd.iterdir()}
  disallowed_dir_objs = curr_dir_objs.difference(ALLOWED_DIR_OBJS)
  if len(disallowed_dir_objs) > 0:
    print(f"Cannot create a scallopy plugin in non-empty directory. Found object `{disallowed_dir_objs.pop()}`")
    exit(2)

  # Collect essential information
  print("This script will guide you creating a new scallopy plugin.")
  print("Please help answer the following questions:")
  name = do_prompt_and_get_answer("Plugin Name: scallop-", pattern=r"^[a-zA-Z][a-zA-Z\-0-9]*$")
  version = do_prompt_and_get_answer("Version (1.0.0): ", pattern=r"^\d+\.\d+\.\d+$", require_non_empty=False, default_value="1.0.0")
  assert name is not None and version is not None

  # Run git init
  if not is_git_repo(cwd):
    subprocess.run(["git", "init", "--quiet"], cwd=cwd, check=True)

  # Write .gitignore file
  (cwd / ".gitignore").write_text(GITIGNORE_SRC, encoding="utf-8")

  # Create pyproject.toml
  underscore_name = name.replace("-", "_")
  plugin_name = "".join([part.title() for part in name.split("-")])
  py_project_src = PY_PROJECT_SRC.format(dashed_name=name, underscore_name=underscore_name, plugin_name=plugin_name, version=version)
  (cwd / "pyproject.toml").write_text(py_project_src, encoding="utf-8")

  # Create makefile
  (cwd / "makefile").write_text(MAKEFILE_SRC.format(dashed_name=name), encoding="utf-8")

  # Create folders
  package = cwd / "src" / f"scallop_{underscore_name}"
  package.mkdir(parents=True, exist_ok=True)
  (package / "__init__.py").write_text(INIT_SRC.format(plugin_name=plugin_name), encoding="utf-8")

  # Final prints
  print(DONE_PRINT.format(path=cwd.absolute()))
