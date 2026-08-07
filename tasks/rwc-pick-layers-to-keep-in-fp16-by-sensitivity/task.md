## Context

In mixed‑precision training, the key‑value (KV) tensors of transformer layers are often quantized to FP8 for memory efficiency.  
The *sensitivity* of a layer can be quantified by the error introduced when its KV tensor is stored in FP8 instead of FP16.  A larger error indicates that the layer would benefit more from retaining FP16 precision.

Let $e_i$ denote the measured FP8‑KV error for layer $i$.  
Given an array $\mathbf{e} = (e_0, e_1,\dots,e_{n-1})$, we wish to select the top–$k$ layers with the largest errors so that they are kept in FP16.

## Task

Implement `select_fp16_layers`:

```python
def select_fp16_layers(errors: list[float], k: int) -> list[int]:
    ...
```

* `errors`: a list of floats of shape $(n,)$ containing the per‑layer errors.  
* `k`: the number of layers to keep in FP16 ($0 \le k \le n$).  

The function must return a **list** of layer indices sorted by decreasing error (ties broken by increasing index).  The returned list should contain exactly $k$ integers.

## Example

```python
errors = [0.02, 0.15, 0.07, 0.15, 0.01]
k = 3
indices = select_fp16_layers(errors, k)
print(indices)   # [1, 3, 2]
```

The two layers with error $0.15$ are chosen first (index $1$ before $3$ because of the tie‑break rule), followed by the layer with error $0.07$.

## What the gate checks

A single gate named `exact_match` verifies that the list returned by your implementation matches exactly the reference list computed by a Python oracle.  No other metrics are evaluated.
