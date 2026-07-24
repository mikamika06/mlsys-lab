## Context

Kernel-fusion compilers (XLA, TVM, TensorRT, `torch.compile`) decide, for every
producer→consumer edge in a computation graph, whether the two ops can be
emitted as **one** kernel launch or must be materialised as two separate
launches with an intermediate tensor written to and read back from memory.

The decision depends on the *kind* of each op:

* **`ew`** (elementwise / pointwise) — `add, mul, sub, div, relu, gelu, sigmoid, silu`.
  Each output element depends on only the corresponding input element(s), so
  the iteration space is trivial to share with a neighbour.
* **`reduce`** (reduction-shaped) — `sum, mean, max, softmax, layernorm`.
  Collapses (part of) an axis; changes the iteration space.
* **`gemm`** (matmul-shaped) — `matmul, conv2d, linear`.
  Has its own tiling/blocking schedule, independent of any neighbour's.
* **`reshape`** (free metadata op) — `reshape, transpose, view, permute`.
  Touches no data, only strides/shape metadata, so it is *always* free to
  fuse into whatever sits next to it.

Given the kind of a producer op and the kind of its consumer, the fusion rule
table is:

| producer \ consumer | `ew` | `reduce` | `gemm` |
|---|---|---|---|
| `ew` | fuse | fuse | **no** |
| `reduce` | fuse | **no** | **no** |
| `gemm` | fuse | **no** | **no** |

(Reading a row: an elementwise producer fuses into an elementwise or a
reduction consumer, but *not* into a gemm consumer — a gemm has its own
tiling and cannot simply absorb an arbitrary elementwise loop upstream. A
gemm *producer*, on the other hand, fuses into a following elementwise op —
this is the classic "bias-add + activation epilogue" fused into the GEMM
kernel. Two reductions, or any pair involving two gemms, or a
reduction feeding a gemm, never fuse.)

`reshape`/`transpose`/`view`/`permute` fuse with **anything**, in either
position, because they cost no compute.

## Task

Implement:

```python
def classify_fusable(op_pairs: list[tuple[str, str]]) -> list[bool]:
    ...
```

* `op_pairs` — a list of `(producer_op_name, consumer_op_name)` string pairs.
  Every op name is one of:
  `add, mul, sub, div, relu, gelu, sigmoid, silu` (kind `ew`),
  `sum, mean, max, softmax, layernorm` (kind `reduce`),
  `matmul, conv2d, linear` (kind `gemm`),
  `reshape, transpose, view, permute` (kind `reshape`).

For each pair, decide whether the producer op can be fused directly into the
consumer op's kernel using the rules above, and return one `bool` per input
pair, **in the same order** as `op_pairs`.

## Example

```python
classify_fusable([("relu", "add"), ("sum", "relu"), ("matmul", "matmul"),
                   ("matmul", "relu"), ("transpose", "sum")])
# -> [True, True, False, True, True]
```

`("relu", "add")`: `ew`→`ew`, fuses.
`("sum", "relu")`: `reduce`→`ew`, fuses.
`("matmul", "matmul")`: `gemm`→`gemm`, does not fuse.
`("matmul", "relu")`: `gemm`→`ew`, fuses (bias/activation epilogue).
`("transpose", "sum")`: `reshape` is free, always fuses.

## What the gate checks

A single gate, **exact_match**, builds a seeded random list of 40 op-name
pairs, computes the ground-truth fusability label for every pair from the
op-kind table and the fusion-rule table above, and compares your returned
list to it element by element. Any mismatch — wrong label, wrong order,
wrong length, or an exception — fails the gate.
