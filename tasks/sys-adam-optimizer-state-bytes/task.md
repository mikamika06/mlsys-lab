## Context
When training neural networks with the Adam optimizer, the optimizer maintains state variables for each parameter to adapt the learning rate. Specifically, it tracks the first moment (moving average of the gradient, $m$) and the second moment (moving average of the squared gradient, $v$).

In standard single-precision (FP32) training, both $m$ and $v$ are stored in FP32, requiring 4 bytes each per parameter. Thus, the optimizer state overhead is 8 bytes per parameter.

In mixed-precision training (FP16 or BF16 weights), the optimizer still maintains $m$ and $v$ in FP32. Furthermore, a high-precision FP32 "master copy" of the weights is required for reliable weight updates, because the updates are often too small to be represented in FP16/BF16. This adds another 4 bytes per parameter to the optimizer state, resulting in 12 bytes of optimizer state per parameter.

## Task
Write a function `adam_optimizer_state_bytes(num_params: int, mixed_precision: bool) -> int` that calculates the total memory in bytes required for the Adam optimizer state.

- `num_params`: The total number of parameters in the model.
- `mixed_precision`: A boolean indicating if mixed-precision training (with an FP32 master weight copy) is being used.

The function should return the exact number of bytes.

## Example
```python
# A model with 1 million parameters in standard FP32 training
bytes_standard = adam_optimizer_state_bytes(1_000_000, False)
# Returns: 8000000 (8 MB)

# A model with 1 million parameters in mixed-precision training
bytes_mixed = adam_optimizer_state_bytes(1_000_000, True)
# Returns: 12000000 (12 MB)
```

## What the gate checks
- The gate computes the `size_ratio` between your returned byte count and the expected true byte count.
- The `size_ratio` must equal $1.0$ exactly across multiple test cases.
