from scallopy import ScallopContext, modes

# Create new context
ctx = ScallopContext.from_mode(modes.min_max_prob)

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0.1, (0, 1)), (0.2, (1, 2)), (0.3, (2, 3))])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
paths = ctx.relation("path")
for row in paths:
  s, t = row.tuple
  print(f"prob: {row.value}, elem: ({s}, {t})")
