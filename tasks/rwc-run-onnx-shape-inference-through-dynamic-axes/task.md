## Context

An ONNX-style graph compiler never runs a model just to learn the shapes of its
tensors. Instead it performs **static shape inference**: it walks the graph
once, and for every node it derives the output tensor's shape purely from the
shapes of its inputs and the node's attributes. This must work even when some
axes are *dynamic* — represented not by an integer but by a named symbol such
as `"N"` (e.g. a batch dimension that is unknown until run time).

We restrict to four ops, each with its own shape rule (`A`, `B`, ... denote
input shapes as tuples/lists; an entry is either a Python `int` or a symbol
`str`):

* **MatMul** — inputs `[A, B]`. Both are rank 2 `(m, k) x (k, n) -> (m, n)`,
  or both are rank 3 with a leading batch axis `(batch, m, k) x (batch, m', k')
  -> (batch, m, k')` where `batch` is carried over from `A` unchanged (the two
  inputs' batch axes are always identical in the graphs you'll be given).

* **Reshape** — inputs `[A]`, attribute `"shape"` is a target shape list. Each
  entry `t` at position `i` produces output dim `i` as follows: if `t == 0`,
  copy `A[i]` (ONNX's "inherit this axis from the input" sentinel — this is
  how a dynamic axis flows through a reshape); otherwise the output dim is
  `t` itself (a concrete int or a symbol).

* **Concat** — inputs `[X1, X2, ...]` (all the same rank), attribute `"axis"`.
  Every dim except `axis` is copied unchanged from the first input. The
  `axis` dim is the *symbolic sum* of the corresponding dim across all
  inputs, combined with this canonicalisation:
  - sum all concrete-int entries into a single integer `total`;
  - for the symbol entries, count repeats with a `Counter`; a symbol `s`
    appearing `n` times contributes the term `s` if `n == 1`, else `"{n}*s"`;
  - emit `total` (an `int`) if there are no symbol terms; otherwise join the
    symbol terms **sorted alphabetically**, and append `str(total)` as a
    final term only if `total != 0`, all joined with `"+"`.
  - Example: entries `[3, "N", 4, "N"]` -> ints sum to `7`, symbol `"N"`
    appears twice -> `"2*N+7"`. Entries `["M", "N"]` -> `"M+N"`. Entries
    `["N", "N", "N"]` -> `"3*N"`.

* **Gather** — inputs `[Data]`, attributes `"axis"` and `"indices_shape"`
  (the shape of the (integer) indices tensor — Gather doesn't need the
  indices' *values* to infer the output shape, only their shape). Output is
  `Data[:axis] + indices_shape + Data[axis+1:]` — the gathered axis is
  spliced out and replaced by the indices tensor's own shape, which can
  change the output's rank.

## Task

Implement `infer_shapes`:

```python
def infer_shapes(input_shapes: dict, graph: list) -> dict:
    ...
```

* `input_shapes` — `dict[str, list]` mapping each graph-input tensor name to
  its shape (a list whose entries are `int` or `str` symbols).
* `graph` — an ordered list of node dicts, each
  `{"name": str, "op": str, "inputs": [str, ...], "attrs": {...}}` where
  `op` is one of `"MatMul"`, `"Reshape"`, `"Concat"`, `"Gather"`, `inputs`
  names tensors that are either graph inputs or the `name` of an earlier
  node in the list, and `attrs` holds the op-specific attributes described
  above.

Process the nodes **in order** (a node's inputs are always already known)
and return a `dict` mapping every node's `name` to its inferred output shape
(a list of `int`/`str` entries). You do not need to include the graph inputs
in the returned dict.

## Example

```python
input_shapes = {"x": ["N", 8], "w": [8, 4]}
graph = [
    {"name": "mm", "op": "MatMul", "inputs": ["x", "w"], "attrs": {}},
    {"name": "rs", "op": "Reshape", "inputs": ["mm"], "attrs": {"shape": [0, 4]}},
]
infer_shapes(input_shapes, graph)
# -> {"mm": ["N", 4], "rs": ["N", 4]}
```

## What the gate checks

A single **exact_match** gate builds several small graphs mixing all four ops
(including cases where a Concat axis mixes multiple distinct symbols, or the
same symbol repeated, with concrete ints) and compares your returned shape
for every node against a reference implementation of the rules above,
dimension by dimension (values are compared numerically when both sides are
integer-like, otherwise as strings). Any mismatch, wrong rank, missing node,
or exception fails the gate.
