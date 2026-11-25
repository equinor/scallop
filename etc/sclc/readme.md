# Scallop Compiler

The Scallop Compiler `sclc` can be used in two different ways, a library or executable.

The library is `sclc_core` and should be used the following way from Rust:

``` rs
use sclc_core::exec::create_executable;
create_executable(&opt, compile_options, &ram)?;
```

The executable `sclc` is invoked in command line.
It takes in a Scallop file (`.scl`) and generates an executable by default.
The executable can be invoked directly.

``` sh
$ sclc program.scl
$ ./program
```

Rust generation compiles a fixed set of rules. It supports Scallop's built-in
operations, including ranges, aggregates, algebraic data types, and CSV input
and output. Custom foreign functions, predicates, aggregates, Python
attribute processors, and external tensor providers are not supported. Use
the interpreter for programs that register extensions or change their rules
at runtime.

Generated Python batches run in input order when the program writes a CSV
file, so the final batch item determines the file contents.

The `scallop!` Rust macro also accepts a raw string literal, which is useful
for Scallop literals such as `t"2020-01-31"` that Rust cannot tokenize directly.
Generated Rust exposes `try_run` and `try_run_with_edb` for fallible execution;
the latter also accepts runtime options such as the scheduler and iteration
limit.
