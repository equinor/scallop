# `scallopy`, the Python binding for Scallop

## Quick Start

With [uv](https://docs.astral.sh/uv/getting-started/installation/) and Rust installed,
you can build and install Scallopy from GitHub:

``` bash
uv venv --python 3.14
uv pip install "scallopy @ git+https://github.com/equinor/scallop.git#subdirectory=etc/scallopy"
```

## Sample usage

``` python
from scallopy import ScallopContext

# Create new context
ctx = ScallopContext()

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2), (2, 3)])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
paths = ctx.relation_tuples("path")
for (s, t) in paths:
  print(f"elem: ({s}, {t})")
```

## Build and Use

Run these commands from the repository root. CI tests CPython 3.11 through 3.15.

``` bash
uv venv --python 3.14
make install-scallopy
uv run --no-project python etc/scallopy/examples/edge_path.py
```

To develop and run the Python tests, sync the locked dependencies into `.venv`
(including CPU PyTorch on Linux), then build the editable package and run the tests:

``` bash
UV_PYTHON=3.14 make init-venv
make test-scallopy
```

Run `make lint-python typecheck-python` from the repository root to check the
library and native extension stubs, CLI, and all plugins with Ruff and `ty`.
Both checks match CI, use tools locked by `uv`, and target Python 3.11.

Install experiment dependencies as needed; they are not part of the test lockfile.
For example, the MNIST experiment uses PyTorch, TorchVision, and `tqdm`:

``` bash
uv pip install torch torchvision tqdm
uv run --no-project python experiments/mnist/sum_2.py
```

## Documentation

Please check out the following file [`scallopy/scallopy.pyi`](scallopy/scallopy.pyi) for documentation.
