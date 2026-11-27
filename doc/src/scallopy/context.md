# Scallop Context

The most fundamental point of interaction of `scallopy` is `ScallopContext`.
The following is a very simple example setting up a `ScallopContext` to compute the `edge-path` program:

``` py
import scallopy

# Creating a new context
ctx = scallopy.ScallopContext()

# Add relation of `edge`
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2)])

# Add rule of `path`
ctx.add_rule("path(a, c) = edge(a, c) or path(a, b) and edge(b, c)")

# Run!
ctx.run()

# Check the result!
print(list(ctx.relation_tuples("path"))) # [(0, 1), (0, 2), (1, 2)]
```

Roughly, the program above can be divided into three phases:

1. Setup the context: this involves defining relations, adding facts to relations, and adding rules that do the computation
2. Running the program inside of context
3. Fetch the results

While the 2nd and 3rd steps are the place where the computation really happens, it's more important for the programmers to correctly setup the full context for computation.
We now elaborate on what are the high-level things to do when setting up the context

## Configurations

When creating a new `ScallopContext`, one should configure it with intended provenance.
If no argument is supplied, as shown in the above example, the context will be initialized with the default provenance, `unit`, which resembles untagged semantics (a.k.a. discrete Datalog).
To explicitly specify this, you can do

``` py
ctx = scallopy.ScallopContext.from_mode(scallopy.modes.unit)
```

Of course, Scallop can be used to perform reasoning on probabilistic and differentiable inputs.
For instance, you can write the following

``` py
ctx = scallopy.ScallopContext.from_mode(scallopy.modes.min_max_prob)
# or
ctx = scallopy.ScallopContext.from_mode(scallopy.modes.diff_min_max_prob)
```

For more information on possible provenance information, please refer to the [provenance](scallopy/provenance.md) section.
It it worth noting that some provenance, such as `topkproofs`, accept additional parameters such as `k`.
Pass those options when selecting the mode:

``` py
ctx = scallopy.ScallopContext.from_mode(scallopy.modes.top_k_proofs(k=5))
```

### Typed modes and result rows

Use `ScallopContext.from_mode()` to have a type checker infer the accepted fact
format and the result value type from the chosen provenance. Both `relation()`
and `relation_rows()` yield named rows with `tuple` and `value` fields. Batch
results use the same row shape:

```py
ctx = scallopy.ScallopContext.from_mode(scallopy.modes.top_k_proofs(k=5))
ctx.add_relation("digit", int)
ctx.add_facts("digit", [(0.8, (2,)), (0.2, (3,))])
ctx.run()

for row in ctx.relation("digit"):
  print(row.tuple, row.value)  # tuple and float
```

In a context that normally takes tagged facts, use `add_untagged_facts()` for
relations declared with `non_probabilistic=True`. The relation schema still
determines the allowed contents of each fact tuple at runtime.

| Mode | `row.value` |
|---|---|
| `modes.unit` | `None` |
| `modes.proofs` | A list of proofs, each represented by a list of fact IDs |
| `modes.tropical` | `int` |
| `modes.top_k_proofs(...)` | A `float` score based on the retained proofs, not necessarily the full query probability |
| `modes.diff_top_k_proofs(...)` | A PyTorch tensor that supports backpropagation |
| `modes.diff_top_k_proofs_debug(...)` | A pair containing a PyTorch tensor and its proof details |

To migrate from bare tuples, use `ctx.relation_tuples("digit")`, or call
`.tuples()` on a batch result collection. To migrate from `(value, tuple)` pairs,
use `row.value` and `row.tuple`. Proof rows now expose their proofs in `row.value`.

`Module(provenance=modes.diff_top_k_proofs(k=5), ...)` also accepts typed modes.
Use `ctx.clone(modes.top_k_proofs(k=5))` to change to a compatible provenance
with explicit options, or `ctx.clone()` to preserve the current mode and options.
Direct string selections are deprecated; use `modes.from_name(name, k=5)` when
the identifier comes from a CLI argument or configuration.

Top-k proof and top/bottom-k clause modes use disjunction groups to prune
conflicting alternatives, then score the retained formula using independent
fact weights. Use `anytime_bounds` when disjunction groups should represent
categorical draws.

### Anytime probability bounds (experimental)

For a finite program with weighted facts and stratified negation, use the
`anytime_bounds` mode and request one ground query instead of calling `run()`:

```py
ctx = scallopy.ScallopContext.from_mode(scallopy.modes.anytime_bounds)
ctx.add_relation("witness", int)
ctx.add_facts("witness", [(0.6, 1), (0.6, 2)])
ctx.add_rule("hit() = witness(x)")

row = ctx.evaluate_bounds("hit", (), tolerance=0.01, max_facts=32)
print(row.value.lower, row.value.upper)  # encloses 0.84
```

The evaluator progressively makes more uncertain facts symbolic. At each
checkpoint, the remaining facts start with a false lower formula and a true
upper formula. Negation complements and swaps these formulas. The evaluator
runs each stratum to a fixed point, then counts the query formulas.
This keeps shared facts correlated across proofs and encloses the **full**
query probability, not a top-k score. A ground query with no possible proof
still returns an explicit interval.

Use `evaluate_relation_bounds("witness")` to get one interval per tuple in
a relation. Each checkpoint evaluates the relation once, sharing the
symbolic fact budget across all rows. It stops at tolerance only when every
returned row meets it, and omits tuples whose upper bound is zero. Rows
with positive upper bounds can still have true probability zero. Whole
relations support `"input"` and `"weight"` fact order; use ground queries
for `"required"` or `"lookahead"`.

Use `evaluate_bounds_batch([("hit", ()), ("witness", (1,))])` for a list of
ground queries, or `evaluate_bounds_batch_with_gradients` for their
derivatives. Results follow query order, including duplicates. Each
query has its own budget. As with `run_batch`, an optional `inputs` mapping
can supply one fact list (or `(facts, disjunctions)` pair) per query for
each input relation. These facts supplement context facts for that example
only. Without example-local inputs, queries over the same relation reuse
the prepared input snapshot and identical queries reuse their results.
Distinct ground tuples still refine independently.

File-backed relations can use the standard CSV input options, including
`has_probability=True` for a leading probability column. Each weighted
CSV fact is independent; CSV input does not declare categorical groups.

`max_facts` caps the number of uncertain facts made symbolic, not the elapsed
time or the number of proofs. Only relations that can affect the query are
evaluated. At each checkpoint, inference can stop on
`tolerance`, `fact_limit`, or `complete` (all uncertain facts considered).
`active_facts` records how many facts were made symbolic. Even on completion,
outward floating-point rounding can leave a small interval instead of a
single number. The prototype recomputes symbolic results at checkpoints; it
does not implement the persistent decision-diagram evaluation of
[Tp-compilation](https://ijcai.org/proceedings/2015/263).

To extend a ground probability query's budget without restarting it:

```py
session = ctx.start_bounds("hit", (), tolerance=0.0)
first = session.advance(max_facts=1)
second = session.advance(max_facts=2)
```

A session snapshots the program, facts, tensor weights, and any CSV inputs
when started. Later changes are visible only to new sessions. Each
`advance()` takes a **total** fact budget; it may stay the same or increase.
Completed checkpoints aren't replayed, and their certificates are retained,
so bounds only tighten. A budget can add a checkpoint that a one-shot query
would skip. A session that reaches its tolerance or completes
returns the same certificate on later calls. New checkpoints still run a
fresh fixed point. Sessions are in-memory and return probability bounds only.

Pass `timeout=` in seconds to `advance()` to bound the time spent refining.
On `time_limit`, the session returns the latest completed certificate (or
`[0, 1]` if none finished) and can resume with another call. The deadline is
cooperative, not a strict wall-clock guarantee: it is checked between rule
updates and around weighted model counting, but a single update or count can
overrun it. It does not include the initial snapshot taken by `start_bounds()`.

For positive queries over independent facts, `fact_order="required"`
prioritizes facts needed by every proof, then ranks the remaining facts by
descending weight. `fact_order="weight"` uses descending weight alone.
Both preserve the original order of ties. The default is `"input"`; queries
involving negation, aggregation, or exclusive groups use input order even
when ranking is requested. Ranking changes which facts fit in the budget,
not the probability being bounded.

`fact_order="lookahead"` starts with required-fact order. At the four-fact
checkpoint (or at the requested partial budget when computing gradients),
it can try up to two replacements for the newest fact, keeping the prefix
that produces the narrowest probability interval. It skips required facts,
budgets below four, and queries that already meet their tolerance. Each
probe respects the fact cap, but probes take extra time and do not optimize
gradient interval width.

For positive queries over independent facts, `checkpoint_schedule="adaptive"`
keeps the zero- and one-fact checkpoints, then quadruples the active prefix
when the previous checkpoint did not tighten either endpoint. It never
exceeds `max_facts`, but can pass a checkpoint that would have met the
tolerance with fewer facts. The default is `"doubling"`; lookahead,
negation, aggregation, and exclusive groups always use doubling.

You can also use PyTorch tensors as independent fact weights:

```py
import torch

ctx = scallopy.ScallopContext.from_mode(scallopy.modes.anytime_bounds)
ctx.add_relation("witness", int)
weight = torch.tensor(0.6, requires_grad=True)
ctx.add_facts("witness", [(weight, 1), (0.25, 2)])
ctx.add_rule("hit() = witness(x)")

result = ctx.evaluate_bounds_with_gradients("hit", ()).value
print(result.gradients[0].lower, result.gradients[0].upper)  # encloses 0.75
result.probability.backward()
```

By default, `evaluate_bounds_with_gradients()` refines through its fact
budget (unlimited unless specified) and reports one certified derivative interval per fact whose
weight requires gradients. Reusing the same tensor for multiple independent
facts gives multiple intervals; autograd sums those contributions. If the
query formula is resolved, `probability` is a scalar with first-order autograd
support. Otherwise `probability` is `None`, even if the probability interval
is narrow. The derivative intervals bound the full query probability's
partials, not derivatives of the interval endpoints.

To stop as soon as selected gradients are certified to a given width:

```py
result = ctx.evaluate_bounds_with_gradients(
    "hit", (),
    gradient_tolerances=[(weight, 0.02)],
    probability_tolerance=0.01,
    max_facts=32,
).value
```

The query stops with `tolerance` when every requested interval is narrow
enough at the same checkpoint. A tensor selects every fact weight that
uses that exact tensor object. If you omit `probability_tolerance`, only
the requested gradient widths govern early stopping. The result always
includes bounds for all trainable fact weights. If the fact cap is reached
first, it returns `fact_limit` with the valid intervals so far. At `complete`,
rounding can still prevent an interval from meeting a very small tolerance.
Meeting a tolerance does not by itself enable exact autograd.

To backpropagate before resolution, explicitly ask for an approximation:

```py
result = ctx.evaluate_bounds_with_gradients("hit", (), max_facts=1).value
approx = result.approximate_backward()
approx.estimate.backward()
```

`approx.estimate` uses the probability interval midpoint for its value and
the gradient interval midpoints for its first-order backward pass. It is
separate from `result.probability`, which is still `None` when unresolved.
`approx.absolute_error` bounds the probability error. Each entry in
`approx.gradients` reports the coefficient used and its absolute error bound
for that fact weight, including rounding to the input tensor's dtype. These
are per-fact partial bounds, not certificates for gradients after PyTorch
applies the chain rule, scales the incoming gradient, or sums contributions
from multiple facts that reuse a tensor.

For a resolved ground query, pass `second_order=True` to
`evaluate_bounds_with_gradients()` and use `torch.autograd.grad(...,
create_graph=True)` for double backward. It differentiates the resolved
possible-world probability, including normalized categorical weights. The
Hessian uses binary64 arithmetic and is not an interval certificate.
Unresolved queries still have no exact scalar. The opt-in computation can
require quadratically many weighted counts in the number of differentiated
facts; derivatives above second order are not supported.

`evaluate_relation_bounds_with_gradients("relation")` obtains these same
first-order results for all rows from one symbolic execution at the fact budget.
It retains rows with possible nonzero sensitivity even when their current
probability is certified zero. Like `evaluate_relation_bounds()`, it supports
`fact_order="input"` or `"weight"`; the order and budget are shared by all rows.

For an independent fact, this partial is the difference between the query
probabilities with the fact forced true and forced false, even at weights of
zero or one. For a monotone query this is its Birnbaum importance: the
probability that changing the fact changes the answer. The gradient API
can bound this score before resolution. Lookahead does not compute exact
importance to rank facts; it measures probability interval width directly.

Input facts can be grouped with `add_facts(..., disjunctions=[[0, 1]])` or
with semicolon-separated alternatives in Scallop source. Each group is one
categorical choice, independent of other groups and independent facts. By
default its supplied weights are the probabilities of the alternatives, and
the remaining mass is the probability that none occurs. A group whose
represented weights sum to more than one is invalid. If exactly one
alternative must occur, pass `normalize_disjunctions=True` to either query
method to normalize the weights within each group. Each normalized group
must have positive total weight.

Groups become symbolic together. A group consumes one slot in `max_facts`
per supplied alternative, including zero-weight alternatives. A budget may
stop short of the next whole group. Gradient intervals are for the supplied
weights: increasing a weight takes mass from the "none" outcome by default,
or changes all normalized outcome probabilities when normalization is
enabled. Using PyTorch weights derived from logits preserves the chain rule.

Use `add_joint_choice()` when one outcome should make several facts true
together, possibly across relations:

```py
ctx.add_relation("alarm", int)
ctx.add_relation("sensor", int)
ctx.add_joint_choice([
    (0.25, {"alarm": [1], "sensor": [1]}),
    (0.50, {"sensor": [2]}),
    (0.125, {}),
])
```

The first outcome makes both `alarm(1)` and `sensor(1)` true with probability
0.25; they are not separate draws. Each outcome can assert multiple rows or
none. Joint choices use the same weight, optional normalization, and gradient
rules as other input groups. A group costs one symbolic fact per outcome,
not per asserted row, and is considered after ordinary input facts.
Different joint choices are independent; encode a
dependency in a single joint distribution. This API does not infer or factor
arbitrary conditional probability tables.

Disjunctive rule heads in source passed to `add_program()` can also generate
categorical choices:
`rel { face(x, 1); face(x, 2); face(x, 3) } = rolled(x)` selects exactly
one distinct head tuple uniformly for each `x` where `rolled(x)` holds.
Different source rules and groundings select independently. Multiple
proofs, alternate body branches, and recursive revisits use the same
choice, not new draws. A rule with no true body produces no head tuple.
This works with stratified negation; head alternatives must use the same
variable pattern. Rule choices are always exhaustive, irrespective of
`normalize_disjunctions`, and do not introduce trainable weights.
The evaluator first discovers possible rule groundings without making
uncertain facts symbolic. Their choices are budgeted after input facts,
with one slot per alternative; a single distinct alternative is deterministic.

`count` counts distinct tuples in each possible world and can be used in
comparisons such as `n >= 2`. An implicit group appears only when at least
one member exists in that world. An explicit `where` group can return zero
when its domain row exists but no members do. `min`, `max`, `argmin`,
`argmax`, `exists`, `sum`, and `prod` also operate per possible world.
Extrema return no result for an empty group, and argument extrema retain
all tied arguments. `sum` and `prod` return zero and one, respectively,
for empty ungrouped or explicit groups. Implicit groups with no members
are absent, even for a reducer that has an identity. Integer sums and
products wrap at the width of their type; floating-point reducers fold
in tuple order. Many distinct numeric results can make a checkpoint
expensive even with a small symbolic fact budget.
`forall` also works ungrouped or with an explicit `where` domain.

Nonrecursive rules can compute values with comparisons, Boolean operators,
conditionals, addition, and subtraction. Addition and subtraction are
allowed only when the query slice has no date/time or duration values.
Filters use the same operator subset, including in recursive rules.
Computed projections in recursive strata and unchecked arithmetic and casts
are not supported. Neither are `!` snapshot variants, score-based or sampled
aggregates, or other reducers outside the list above.

Bounds queries can use these deterministic built-in functions:
`floor`, `ceil`, `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `tan`, `asin`,
`acos`, `atan`, `string_concat`, `string_length`,
`string_char_at`, `string_index_of`, `string_upper`, `string_lower`,
`string_replace`, `string_trim`, `datetime_day`, `datetime_month`,
`datetime_month0`, and `datetime_year`. If a function has no result for
its arguments, that computation produces no row, as in ordinary execution.
Supported untagged predicates are `float_eq`, `range`, `string_chars`,
`string_find`, `string_index`, `string_split`, and `datetime_ymd`. Only
`float_eq` can run within a recursive stratum: the other predicates
generate values. Generated rows inherit their input provenance and do not
consume additional `max_facts` slots. A large generated range can still
make a checkpoint expensive. Custom callbacks and tagged or stateful
built-ins are not supported.

`@goal` does not trigger early termination in bounds mode. Checkpoints
always run to the least fixed point. `@scheduler("lfp")` (also `"bfs"`)
is supported; other scheduler annotations are rejected. Indexed storage
and joins, provenance-overwriting rules, and entity, symbol, or tensor
tuple values are also unsupported. Other provenance modes retain their
existing semantics and output types.

## Adding Program

Given that a context has been configured and initialized, we can set it up the quickest by loading a program into the context.
One can either load an external `.scl` file, or directly inserting a program written as Python string.
To directly add a full program string to the context, one can do

``` py
ctx.add_program("""
  rel edge = {(0, 1), (1, 2)}
  rel path(a, c) = edge(a, c) or path(a, b) and edge(b, c)
""")
```

On the other hand, assuming that there is a file `edge_path.scl` that contains the same content as the above string, one can do

``` py
ctx.import_file("edge_path.scl")
```

## Adding Relations

Instead of adding program as a whole, one can also add relations one-at-a-time.
When adding new relations, one would need to supply the name as well as the type of the relation.
For example, the `edge` relation can be defined as follows

``` py
ctx.add_relation("edge", (int, int))
```

Here, we are saying that `edge` is an arity-2 relation storing pairs of integers.
Note that we are specifying the type using Python's `int` type.
This is equivalent to the `i32` type inside Scallop.
Therefore, the above instruction tranlates to the following Scallop code:

``` scl
rel edge(i32, i32)
```

Many existing Python types can directly translate to Scallop type.
In particular, we have the mapping listed as follows:

| Python Type | Scallop Type |
|-------------|--------------|
| `int` | `i32` |
| `bool` | `bool` |
| `float` | `f32` |
| `str` | `String` |

In case one want to use types other than the listed ones (e.g., `usize`), they can be accessed directly using the string `"usize"`, or they can be accessed through predefined types such as `scallopy.usize`.
The example below defines a relation of type `(usize, f64, i32)`:

``` py
ctx.add_relation("my_relation", (scallopy.usize, "f64", int))
```

Specifically for arity-1 relations, users don't need to use a tuple to specify the type.
For instance,

``` py
ctx.add_relation("digit", int)
```

## Adding Facts

The most basic version of adding facts into an existing relation inside of an existing context.
We are assuming that the context has a provenance of `"unit"`.

``` py
ctx.add_facts("edge", [(1, 2), (2, 3)])
```

If the relation is declared to be having arity-1 and that the type is a singleton type instead of a 1-tuple, then the facts inside of the list do not need to be a tuple.

``` py
ctx.add_relation("digit", int)
ctx.add_facts("digit", [1, 2, 3])
```

### Probabilistic Facts (Tagged Facts)

When the Scallop context is configured to use a provenance other than.
If one wants to add facts along with probabilities, they can wrap their non-probabilistic facts into tuples whose first element is a simple probability.
For example, if originally we have a fact `1`, wrapping it with a corresponding probability gives us `(0.1, 1)`, where `0.1` is the probability.

``` py
ctx.add_facts("digit", [1, 2, 3])                      # without probability
ctx.add_facts("digit", [(0.1, 1), (0.2, 2), (0.7, 3)]) # with probability
```

Of course, if the original facts are tuples, the ones with probability will be required to wrap further:

``` py
ctx.add_facts("color", [("A", "blue"), ("A", "green"), ...])               # without probability
ctx.add_facts("color", [(0.1, ("A", "blue")), (0.2, ("A", "green")), ...]) # with probability
```

Other modes accept different tag types. For example, tropical provenance uses integer costs:

``` py
ctx = scallopy.Context.from_mode(scallopy.modes.tropical)
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(1, (1, 2)), (3, (2, 3))])
```

### Non-tagged Facts in Tagged Context

## Adding Rules

### Tagged Rules

## Running

## Additional Features

There are more features provided by the `ScallopContext` interface.
We hereby list them for reference.

### Cloning

One can copy a context to create a new context.
The resulting context will contain all the program, configurations, and provenance information.

``` py
new_ctx = ctx.clone()
```

The cloning feature relates to pseudo-incremental computation and branching computation.
We elaborate on this in the [Branching Computation](scallopy/branching.md) section.

### Compiling

### Iteration Count Limit

One can configure the

### Early Discarding

### Obtaining Context Information

### Foreign Functions and Predicates

### Saving and Loading

Please refer to the [Saving and Loading](scallopy/save_and_load.md) section for more information.
