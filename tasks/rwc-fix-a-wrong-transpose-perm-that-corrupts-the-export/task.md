## Context

Export pipelines for ML models often contain layout-changing operators such as `Transpose`. A transpose is described by a permutation $p$ of the input axes. For an input tensor $X$ with shape $(d_0, d_1, \dots, d_{n-1})$, the exported result is

$$
Y = \operatorname{transpose}(X, p)
$$

where output axis $i$ is copied from input axis $p_i$.

A corrupted export may contain the wrong permutation while all tensor values and shapes remain otherwise valid. To repair the graph, a debugging tool can compare the exported output against the trusted reference output and search for the permutation that reproduces the reference.

For a small tensor with $n$ dimensions there are $n!$ possible permutations. A production debugging utility can enumerate candidates because export repair is usually performed on small graph fragments, not full model tensors.

## Task

Implement `fix_transpose_perm(input_tensor, exported_output, torch_reference)`.

The function receives three NumPy arrays:

- `input_tensor`: the tensor before the exported `Transpose`.
- `exported_output`: the output produced by the corrupted exported transpose.
- `torch_reference`: the trusted output from the original framework.

Return a tuple containing the corrected transpose permutation as a tuple of integers.

The returned permutation must satisfy:

```python
np.transpose(input_tensor, returned_perm)
```

being equal to `torch_reference`.

The input tensors will have between 2 and 5 dimensions. You may assume that exactly one permutation matches the reference output.

## Example

```python
import numpy as np

x = np.arange(24).reshape(2, 3, 4)

corrupt = np.transpose(x, (2, 1, 0))
reference = np.transpose(x, (1, 2, 0))

perm = fix_transpose_perm(x, corrupt, reference)

# perm == (1, 2, 0)
# np.transpose(x, perm) matches reference
```

## What the gate checks

The gate computes the oracle permutation by independently enumerating all valid transpose permutations and selecting the one whose NumPy transpose matches the trusted reference.

The `perm_exact` score requires the returned permutation to exactly match the oracle tuple. The `max_abs_err` score re-runs the repaired transpose and checks

$$
\max_i |Y_i - Y_i^{\mathrm{ref}}|
$$

is at most $10^{-12}$.
