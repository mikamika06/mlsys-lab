## Context

NumPy broadcasting defines how arrays of different shapes combine in
elementwise operations. The rules determine the output shape without
data copying.

Given two shapes $\mathbf{s}_a = (a_1, \ldots, a_m)$ and
$\mathbf{s}_b = (b_1, \ldots, b_n)$ where $m \le n$, first left-pad
$\mathbf{s}_a$ with $n - m$ ones to obtain
$\mathbf{s}_a' = (1, \ldots, 1, a_1, \ldots, a_m)$. Then at every
dimension $i \in \{1, \ldots, n\}$ the sizes must satisfy

$$a_i' = b_i \quad\text{or}\quad a_i' = 1 \quad\text{or}\quad b_i = 1.$$

The output size at that dimension is $o_i = \max(a_i',\; b_i)$.

For a chain of $k$ elementwise operations the running shape evolves as

$$\mathbf{s}_0 \;\xrightarrow{\mathrm{op}_1}\; \mathbf{s}_1 \;\xrightarrow{\mathrm{op}_2}\; \cdots \;\xrightarrow{\mathrm{op}_k}\; \mathbf{s}_k$$

where $\mathbf{s}_j = \text{broadcast}(\mathbf{s}_{j-1},\; \mathbf{s}_{\text{op}_j})$
applies the rule above at each step. Note that for elementwise ops
(`add`, `subtract`, `multiply`, `divide`) only the two shapes matter—the
operation name has no effect on the output shape.

## Task

Implement `predict_broadcast_shape(ops)`:

```python
def predict_broadcast_shape(ops: list[tuple[str, tuple[int, ...]]]) -> tuple[int, ...]:
```

`ops` is a non-empty list of `(op_name, shape)` tuples.
The first entry `("init", shape)` sets the initial shape.
Each subsequent `(op_name, shape)` represents an elementwise operation
whose operand has the given shape; `op_name` is one of `"add"`,
`"subtract"`, `"multiply"`, or `"divide"`.

Return the final output shape as a `tuple[int, ...]`.
You may assume every step is shape-compatible (no broadcast errors).

## Example

```python
predict_broadcast_shape([
    ("init", (3, 1)),
    ("multiply", (1, 4)),
    ("add", (2, 1, 1)),
])
# Step 1: broadcast (3,1) with (1,4) → (3, 4)
# Step 2: broadcast (3,4) with (2,1,1) → (2, 3, 4)
# Returns (2, 3, 4)
```

## What the gate checks

One gate: `exact_match`. Ten test chains of varying dimensionality and
length are graded. The reference answer is computed live by the NumPy
broadcasting oracle, so only analytically correct predictions pass.
Importing NumPy and performing the actual operations is a valid
strategy, but the exercise is to internalize the padding and
$\max$-per-dimension rules.
