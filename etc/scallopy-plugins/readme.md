# Scallopy Plugins

This directory contains plugins written for `scallopy` for it to have capabilities interacting with other domain and models.
These include interfacing with GPT, vision models like SAM, multi-modal models like ViLT and CLIP, and so on.

## Install plugins

Plugins are installed independently with `uv`, not included in the repository's
test lockfile. First [create a virtual environment and install Scallopy](../scallopy/readme.md#build-and-use).
From the repository root, use whichever of these targets you need:

``` bash
$ make -C etc/scallopy-plugins develop # to develop all plugins
# or
$ make -C etc/scallopy-plugins install # to install all plugins
# or
$ make -C etc/scallopy-plugins develop-<PLUGIN> # to develop the specific <PLUGIN>
# or
$ make -C etc/scallopy-plugins install-<PLUGIN> # to install the specific <PLUGIN>
```

Here, the difference between `develop` and `install` is that `develop` allows you do change the source code and run it without having to re-install everything.
Therefore, using `develop` is best suited if you are actively developing the plugins.
Otherwise, use `install` so that a concrete wheel (`.whl`) is built and installed into your environment.

Some plugins import `scallop-gpu`, which is not published to PyPI. For example,
to use the CLIP plugin, install both packages into the same environment:

``` bash
make -C etc/scallopy-plugins develop-gpu develop-clip
```

Note that for each installed plugin, the `scallop-ext` extension module will load them automatically, unless you specify that you want to include or exclude certain plugins (based on their names).

## Create new plugins

Please follow these steps to create a new plugin.

1. Create a new folder. If you want to create a new plugin in the source, create a folder under this directory.

2. In the folder you just created, add another file named `pyproject.toml`, and use the following template:

``` toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "scallop-<YOUR-PLUGIN>"
version = "0.0.1"
dependencies = ["<YOUR-DEPENDENCY-1>", "<YOUR-DEPENDENCY-2>"]

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."scallop.plugin"]
<NAME> = "scallop_<YOUR-PLUGIN>:ExamplePlugin"
```

Make sure you replace everything marked in angle brackets `<...>` with proper information.
For example, for the plugin for CLIP, we have the plugin named `scallop-clip`, and the `<NAME>` set to `clip`.
Here, the `<NAME>` will be used for Scallop plugin manager and users to decide wether to include or not include the plugin.
Therefore, try your best to name it well.

3. Under the folder you just created, create a new folder `src/scallop_<YOUR-PLUGIN>`.
Notice that there is an underscore (`"_"`) between `scallop` and your plugin name, instead of a `"-"`.

4. Under this folder, create a file named `__init__.py`. You can then add however many files under the same folder.

5. Export a subclass of `scallopy.Plugin` from `__init__.py`. Override only the hooks that the plugin needs:

- `setup_argparse(self, parser)` to add command-line arguments.
- `configure(self, args, unknown_args)` to consume parsed arguments.
- `load_into_ctx(self, ctx)` to register foreign constructs in the context.

Here, we provide some example of a simple plugin that loads a `key` from command-line arguments so that it can be accessed via a foreign function `$get_key()` inside of Scallop:

``` py
import scallopy


class ExamplePlugin(scallopy.Plugin):
  key = "12345678"

  def setup_argparse(self, parser):
    parser.add_argument("--key", default=self.key)

  def configure(self, args, unknown_args):
    self.key = args.get("key", self.key)

  def load_into_ctx(self, ctx):
    @scallopy.foreign_function
    def get_key() -> str:
      return self.key

    ctx.register_foreign_function(get_key)
```
