# Unreleased

- Scallop now executes programs only through the RAM interpreter. The `sclc` executable, `scallop!` Rust macro, and Scallopy's optional native JIT have been removed. Python callers should omit `jit=True`, `jit_name`, and `recompile` (`jit_recompile` on `ScallopForwardFunction`). The Rust extension, `ctx.compile()`, and parallel batch execution remain available.
- Compiler diagnostics now show source spans in Rust-style snippets, including syntax errors from REPL and in-memory input. REPL snippets use a `REPL:<id>` header.

- Weighted model counting always compiles a decision diagram. It uses OxiDD on native targets and RSDD on WebAssembly, where OxiDD cannot start its manager threads. Probabilities and gradients can differ in their last floating-point digits because the diagram changes the order of arithmetic.
- Differentiable min/max combines equal-score tags by a stable derivative order and propagates changes to the winning gradient. Top-k proofs (including differentiable and debug modes) break equal-weight candidate ties by their literals. These modes assign input fact IDs in predicate order, preserving input order within each relation. Top-k still prunes intermediate results, so moving a pruning point can change the result.
- `DateTime` text uses UTC ISO 8601, e.g. `2021-01-01T00:00:01.1Z`, including in `$format` and Scallopy results. The old `2021-01-01 00:00:01.100 UTC` spelling still parses.
- `Duration` text uses hours and minutes where appropriate (`PT90S` becomes `PT1M30S`, and `P0D` becomes `PT0S`). The calendar prefix for durations containing months is unchanged, and the previous spelling still parses.

Rust API migration:
- The standalone `sdd` crate has been removed. To use your own decision diagram implementation, implement `scallop_core::runtime::provenance::WmcBackend` and pass it to the provenance's `new_with_wmc_backend` constructor. Custom provenance implementations can also use `AsBooleanFormula::wmc_using`. Existing provenance constructors select the default backend.
- Use `NodeLocation::report` or `report_warning` instead of supplying a caret color. Custom `Source` implementations can supply `diagnostic_name` and `first_line_number` instead of `line` and `line_name`.
- `ForeignFunction::infer_return_type` overrides now take `&[Value]` instead of `&Vec<Value>`.
- `FrontContext::analysis` is now an `Rc<Analysis>`. Read it through dereferencing and use `Rc::make_mut` to update it.
- Use `TensorValue::checked_add`, `checked_sub`, and `checked_mul` for shape-checked arithmetic. Incompatible shapes still return `None`.
- Custom `ExternalTensor` implementations no longer implement `as_any`. Use `DynamicExternalTensor::cast` or upcast to `dyn std::any::Any` to downcast tensors. Use `Attribute::get` or `std::any::Any` to downcast compiler attributes.
- `SymbolRegistry` now keeps its storage private. Use `register`, `get_id`, and `get_symbol` to intern and look up symbols; IDs are still assigned in insertion order.
- `Disjunctions` now keeps its storage private. Use `add_disjunction` to register facts and `iter` to visit groups in first-seen order.
- Use a struct literal in place of `compiler::back::Reduce::new`. For a front-end `Reduce` with a source span, use `_Reduce::new(...).with_span(start, end)` instead of `Reduce::new_with_span`.
- Construct reduce query-plan nodes with `HighRamNode::reduce`. The `HighRamNode::Reduce` variant now contains a `Box<Reduce>`; front-end AST constructors and accessors are unchanged except for the `Reduce` constructor noted above.
- The unimplemented dense-matrix Rust storage variant has been removed. Scallop still supports `sorted` and `indexed_vec`.
- Use `Timestamp`'s `Display` implementation in place of `utils::display_date_time`.
- The typed dataflow runtime (`runtime::statics`) has been removed along with code generation. Use `runtime::dynamic` for runtime operators, including aggregation. Scallop programs and the dynamic foreign aggregate API are unchanged.

# v0.2.5, Sep 14, 2024

- Adding `@goal` and `@scheduler` to Scallop: now we have multiple semantics driven by scheduling strategies (BFS, DFS, A-Star, Beam, etc.) and stopping criteria.
- Removing the package `scallopy_ext`; moving everything into `scallopy` and simplifying installation.
- Refactoring Scallopy Plugins. All the existing plugins under `etc/scallopy-plugins` are updated to the new API. Instead of having individual entry-points, now Plugins are programed as a Python class inheriting `scallopy.Plugin`.

Lower Level:
- Improving query planning by prioritizing over demand predicates
- Optimizing RAM by introducing deduplication of update rules
- Adding new provenances, including various tropical semirings for discrete and real-number reasoning modes
- Adding optimizations to Scallop runtime join algorithms
- Multiple bugs fixed

# v0.2.4, Aug 30, 2024

- Rule tags can now be expressions with potential reference to local variables: `rel 1/n::head() = body(n)`
- Allowing for sparse gradient computation inside Scallopy to minimize memory footprint
- Allowing users to specify per-datapoint output mapping inside Scallopy
- Adding `topkproofsdebug` provenance for obtaining the top-k proofs in Scallopy
- Adding destructor syntax so that ADTs can be used in a more idiomatic way
- Unifying the behavior of integer overflow inside Scallop
- Multiple bugs fixed

# v0.2.3, Jun 23, 2024

# v0.2.2, Oct 25, 2023

- Adding `wmc_with_disjunctions` option for provenances that deal with boolean formulas for more accurate probability estimation
- Adding foreign aggregate interface
- Added aggregators such as `avg`, `enumerate`, `normalize`, `weighted_sum`, and `sort`
- Adding syntax sugar for aggregation
- Unknown attribute will now be flagged and reported during compile time
- Fixed Scallopy Extension compatibility issue with Python 3.8

# v0.2.1, Sep 12, 2023

- Democratizing foreign functions and foreign predicates so that they can be implemented in Python
- Adding foreign attributes which are higher-order functions
- Adding `scallop-ext` the extension library and multiple Scallop plugins, including `scallop-gpt`, `scallop-clip`, and so on.
- Fixed multiple bugs related foreign predicate computation
- Adding `count!` aggregator for non-probabilistic operation
- Fixed sum and product aggregator so that they can accept additional argument
- Multiple bugs fixed; performance improved

# v0.2.0, Jun 11, 2023

- Fixing CSV loading and its performance; adding new modes to specify `keys`
- Adding `Symbol` type to the language
- Adding algebraic data types to the language and supports for entities
- Adding tensors to the language which can be accessed from `scallopy`
- Adding `scallop` CLI (command-line interface) with OpenAI plugins for invoking LLMs
- Adding more documentations
- Multiple bugs fixed

# v0.1.9, Apr 24, 2023

- Supporting (partial) disjunctive Datalog
- Fixed custom provenance's default implementations and dispatcher fallback
- Adding new negation syntax to `exists` and `forall` expression
- Multiple bugs fixed

# v0.1.8, Mar 27, 2023

- Add foreign predicates including soft comparisons,
- Add `DateTime` and `Duration` support
- Add input-mapping support for multi-dimensional inputs in `scallopy`
- Fixing floating point support by rejecting `NaN` values
- Adding back iteration-limit to runtime environment
- Add Scallop book repository

# v0.1.7, Jan 12, 2023

- Better integration with Extensional Databases (EDB) and memory optimizations
- Better handling of mutual exclusive facts in probabilistic/differentiable reasoning
- Add categorical-k and top-k samplers
- Add foreign function interface available to Python through `scallopy`
- Add syntax sugars for `forall` and `exists` aggregators

# v0.1.6, Nov 17, 2022

`v0.1.6` introduces many improvements and features
- `scallopy` forward function interface for better ergonomics
- Better provenance context interface (saturation, sampling, etc.)
- Multiple bug fixes on provenance structures
- Better support for magic-set transformations on tagged-semantics

# v0.1.5, August 21, 2022

`v0.1.5` introduces JIT compilation and many more features
- JIT compilation in `scallopy` (`jit=True`)
- `scallopy` forward function can output multiple relations instead of just one
- Add standard library functions such as `string_length`, `hash`, etc.
- Add more provenance structures such as `diffnandmultprob`, `diffaddminprob`, etc.
- Systematic internal monitoring and `--debug-tag` options

# v0.1.4, July 24, 2022

`v0.1.4` introduces multiple bug fixes and new features
- forall aggregator (`forall`)
- implies operator (`=>`)
- topkproofs supporting negation and aggregation
- Multiple bugs fixed

# v0.1.3, June 1, 2022

`v0.1.3` introduces multiple bug fixes and few additional features
- WASM Scallop is now able to be compiled
- Fixed the ordering of the relations in the output
- Multiple bugs fixed

# v0.1.2, May 26, 2022

`v0.1.2` introduces a bug fix in the compiler.

# v0.1.1, May 20, 2022

`v0.1.1` introduces
- the implementation of many more aggregators for different provenances
- bug fixed where min/max aggregators are not functioning correctly
- bug fixed where atoms like `pred(a, a)` might cause compilation issue
- adding customization to termination controlled by provenances
- adding `--query` command line argument to `scli`
- allow for disjunction in many provenances that wasn't allowed to

And many more experimental improvements associated with these changes.
