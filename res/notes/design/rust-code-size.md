# Rust source cleanup

The goal is to maintain less handwritten Rust, not to shrink binaries or minimize the lockfile. Rust API breaks are acceptable for this release. Scallop language behavior, symbol IDs, and Python/WASM support should be preserved unless a change is called out explicitly.

## Cleanup proposals

- [Replace custom downcasts, cloning, and symbol interning](https://github.com/equinor/scallop/pull/29). Upcast compiler attributes and external tensors to `dyn std::any::Any`, and derive `Clone` for concrete trait-object wrappers where that is equivalent to the existing implementation. Use `SymbolRegistry` methods rather than accessing its former public map and vector; the private `IndexSet` keeps insertion-order IDs and returns the same ID when a symbol is registered again.
- [Replace the custom copy-on-write wrapper](https://github.com/equinor/scallop/pull/27). `FrontContext::analysis` is an `Rc<Analysis>`; read through dereferencing and mutate through `Rc::make_mut`. Cloned contexts can diverge independently, and failed compilations must still leave the original context unchanged.
- [Use `annotate-snippets` for source diagnostics](https://github.com/equinor/scallop/pull/23). Its renderer can preserve file and REPL labels, Rust macro line numbers, UTF-8 spans, and end-of-input diagnostics. `codespan-reporting` would need an adapter for Rust macro numbering. Source offset lookup uses `slice::partition_point`, keeping byte columns and selecting the last of duplicate offsets.
- [Derive runtime, database, I/O, and related errors with `thiserror`](https://github.com/equinor/scallop/pull/26). Preserve messages and conversions while exposing wrapped causes through `std::error::Error::source`.

## Jiff

[Scallop delegates time output to Jiff](https://github.com/equinor/scallop/pull/30) but retains its input syntax and duration arithmetic. Calendar months are stored separately from elapsed `SignedDuration` so the two components can have opposite signs. DateTime arithmetic applies the calendar shift in UTC before the elapsed component. Neither `std::time::Duration` nor a direct Jiff type substitution preserves that behavior.

## Weighted model counting

Weighted model counting always compiles a diagram. The native backend uses OxiDD with a thread-local manager; each evaluation still has its own weights and memo table. WebAssembly uses RSDD because OxiDD's managers require native threads. Both backends map sparse fact IDs to dense labels and support owned semiring elements such as sparse gradients.

Applications can implement `WmcBackend` and keep their diagram representation private.

### Historical rollout measurements

[RSDD originally replaced the standalone `lib/sdd` crate](https://github.com/equinor/scallop/pull/20). Each RSDD builder allocates two fixed-size unique tables (131,072 slots each), so startup cost mattered. The previous RSDD path counted constants, literals, flat clauses and formulas with up to eight facts directly.

The prototype matched the previous backend and brute-force enumeration on 520 formulas for probabilities and sparse gradients, including constants, negation, sparse IDs, and a two-fact exclusivity case. The prototype timings below predate the bounded direct-count path. They are release-mode medians from five runs of 10,000 calls with prebuilt proof tags (adjacent-pair clauses, with two alternating groups in the exclusive case). They measure diagram compilation and weighted evaluation, not end-to-end Scallop throughput.

| Recovery path | Facts | Old backend | RSDD prototype |
| --- | ---: | ---: | ---: |
| Top-k | 2 | 2.0 µs | 0.04 µs |
| Differentiable sampled | 2 | 2.5 µs | 0.35 µs |
| Differentiable top-k with exclusivity | 2 | 2.7 µs | 130 µs |
| Top-k | 8 | 37 µs | 146 µs |
| Differentiable sampled | 8 | 40 µs | 152 µs |
| Differentiable top-k with exclusivity | 8 | 678 µs | 215 µs |

In a separate release-mode comparison at eight facts, the direct-count path reduced differentiable recovery from 176 to 23 µs without exclusivity and from 247 to 50 µs with exclusivity, compared with always constructing an RSDD.

Earlier synthetic diagram benchmarks had mixed results at 24 to 25 variables. At 64 to 65 variables, RSDD was 1.8 to 8.3 times faster than the old backend. The pinned Git dependency also brings future-compatibility warnings in `bitflags` 0.8 and `nom` 4.

The published `sddrs` 0.1.4 API provides model counting and enumeration, not a weighted fold or public decomposition traversal for sparse gradients. `biodivine-lib-bdd` exposes BDD nodes, but has a 16-bit variable count and would also change the kind of decision diagram.
