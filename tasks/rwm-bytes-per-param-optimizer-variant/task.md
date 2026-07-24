## Context

The Adam optimizer maintains two auxiliary arrays per parameter, the first‑moment estimate $m$ and the second‑moment estimate $v$. If a model has $n$ parameters, each array contains $n$ elements. The total memory footprint of the state is therefore
$$\mathrm{bytes}_{\text{state}} = n \bigl(\mathrm{bytes}(m) + \mathrm{bytes}(v)\bigr).$$

When the arrays are stored as 32‑bit floating point numbers, each element occupies $4$ bytes and the per‑parameter cost is
$$\frac{\mathrm{bytes}_{\text{state}}}{n} = 8.$$
If the arrays are quantised to 8‑bit unsigned integers, each element costs $1$ byte and the per‑parameter cost becomes $2$. A 16‑bit floating point representation yields a per‑parameter cost of $4$.

The task is to infer which variant was used from the state alone.

## Task

Implement `optimizer_variant(state_dict)`:

```python
def optimizer_variant(state_dict: dict[str, np.ndarray]) -> str:
    ...
```

`state_dict` contains the arrays that make up the Adam state. All arrays have identical shape and contain one value per model parameter. The function must return a string label chosen from

- `"adam_fp32"` – 8 bytes/parameter,
- `"adam_fp16"` – 4 bytes/parameter,
- `"adam_uint8"` – 2 bytes/parameter.

The implementation should compute the bytes‑per‑parameter value and classify it using thresholds that are robust to small floating‑point inaccuracies. Do **not** hardcode a mapping from specific dtypes; instead base the decision on the numeric cost per parameter.

## Example

```python
import numpy as np

n = 5
state_fp32 = {"m": np.arange(n, dtype=np.float32),
              "v": np.arange(n, dtype=np.float32)}
print(optimizer_variant(state_fp32))   # adam_fp32

state_uint8 = {"m": np.arange(n, dtype=np.uint8),
               "v": np.arange(n, dtype=np.uint8)}
print(optimizer_variant(state_uint8))  # adam_uint8
```

## What the gate checks

The grader constructs synthetic state dictionaries for each variant and compares your output to a reference computed by the same numeric logic. The metric `exact_match` must equal 1.0.
