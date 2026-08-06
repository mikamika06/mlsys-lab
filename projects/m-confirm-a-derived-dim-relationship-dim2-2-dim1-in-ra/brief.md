# Dynamic Shapes & Export Constraints

When compiling models with dynamic shapes, the tracer tracks how dimensions relate to each other. For example, when applying sliding window logic, a dimension `s1` might be constrained to be exactly `2 * s0`.

You are implementing a simplified shape verifier for our ML compiler. The downstream kernel generator needs to verify derived relationships between dimensions and ensure that propagated shapes match the expected exports.

Your tasks:

1. **`confirm_relationship(constraints, dim2, dim1)`**:
   Given a dictionary of `constraints` mapping a derived dimension to `(multiplier, base_dim)` (e.g., `{"s1": (2, "s0")}`), return the integer `k` if `dim2 == k * dim1`. You must resolve this transitively (e.g., if `s2 = 3 * s1` and `s1 = 2 * s0`, then `s2 = 6 * s0`). If they are not proportional or do not share a base dimension, return `None`.

2. **`propagate_shapes(ops, inputs, constraints)`**:
   Propagate shapes through a series of reshape/view operations.
   - `inputs` maps tensor names to their initial shape, e.g., `{"x": [(1, "s1"), (1, "s0")]}`.
   - `ops` is a list of view operations: `{"in": "x", "out": "y", "shape": [(1, "s0"), (2, None), (-1, None)]}`.
   - The wildcard `-1` is always represented as `(-1, None)`.
   - You must compute the inferred value for the `-1` dimension based on the total tensor volume (which is conserved).
   - Return a dictionary mapping all tensor names (both inputs and intermediate/outputs) to their shapes expressed entirely in **base dimensions** as a `tuple` of tuples.

*Note: You can assume divisions for `-1` will always cleanly resolve to either an integer constant or a single variable to the first degree.*
