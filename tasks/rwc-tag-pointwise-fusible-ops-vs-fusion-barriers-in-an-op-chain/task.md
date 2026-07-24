## Context

In deep learning compilers, a *pointwise* operation is one that can be applied independently to each element of an input tensor. Examples are addition, multiplication, ReLU and sigmoid. These ops can usually be fused into a single GPU kernel because they have no inter‑element dependencies. In contrast, operations such as matrix multiplication (`mm`), batched matrix multiplication (`bmm`), convolution (`conv`), reductions or `nonzero` require communication between elements and therefore act as *fusion barriers*.

## Task

Implement the function `tag_ops(op_names: list[str]) -> list[bool]`. It receives a list of operation names (strings) that appear in an execution chain. For each name it should return `True` if the op is pointwise‑fusible, otherwise `False`.

The mapping used by the grader is:

- Fusible ops: `"add"`, `"mul"`, `"relu"`, `"sigmoid"`, `"broadcast"`
- Barrier ops: `"mm"`, `"bmm"`, `"conv"`, `"reduction"`, `"nonzero"`

Any name not in the fusible set should be treated as a barrier.

## Example

```python
>>> tag_ops(["add", "mm", "relu"])
[True, False, True]
```

## What the gate checks

The grader computes a reference label vector using the same rule set and compares it element‑wise with your output. The metric `exact_match` is `1.0` only if all labels agree; otherwise it is `0.0`. No other metrics are used.
