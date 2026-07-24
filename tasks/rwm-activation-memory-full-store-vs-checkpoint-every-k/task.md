## Context

In a deep neural network each layer produces an activation tensor that must be kept in memory until the backward pass can use it to compute gradients.  
If we have $L$ layers, a batch size of $S$, and each activation has $H$ elements (e.g. hidden dimension), then storing every activation requires

$$
M_{\text{full}} = L \times S \times H .
$$

Checkpointing is a strategy that reduces memory usage by only saving activations at regular intervals, recomputing the intermediate ones during back‑propagation.  
With a checkpoint interval of $k$ layers we store activations for every $k^{\text{th}}$ layer (including the first and last).  
Let

$$
C = \left\lceil \frac{L}{k} \right\rceil + 1
$$

be the number of stored checkpoints.  
During back‑propagation we must recompute a segment of $k$ layers; at its peak we hold all intermediate activations for that segment in addition to the checkpoint activation.  
Thus the maximum memory used is

$$
M_{\text{ckpt}} = C \times S \times H + k \times S \times H .
$$

## Task

Implement a function

```python
def compute_peak_activation_memory(L: int, S: int, H: int, k: int) -> tuple[int, int]:
    ...
```

that returns a two‑element tuple `(full_mem, ckpt_mem)` where `full_mem` is the memory required if all activations are stored and `ckpt_mem` is the peak memory when checkpointing every $k$ layers.  
All arithmetic should be performed with integer types; the result must be exact.

## Example

```python
>>> compute_peak_activation_memory(10, 32, 128, 5)
(40960, 32768)
```

Explanation:  
- Full store: $10 \times 32 \times 128 = 40960$  
- Checkpoints: $\lceil 10/5\rceil + 1 = 3$, so $C=3$.  
  Peak memory: $(3+5)\times 32\times 128 = 32768$.

## What the gate checks

The grader will compute a reference value using NumPy arithmetic and compare it to your output.  
Your function must return exactly the same tuple of integers for all test cases; otherwise the `exact_match` gate fails.
