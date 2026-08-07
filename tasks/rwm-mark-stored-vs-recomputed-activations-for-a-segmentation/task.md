## Context

In many deep‑learning training pipelines, checkpointing is used to trade compute for memory.
A *checkpoint boundary* stores the activations of a layer so that later gradients can be computed without re‑running the forward pass.  
Between two boundaries the intermediate activations are **recomputed** during back‑propagation.

Given a sequence of $L$ layers and a segmentation of this sequence into contiguous blocks,
the goal is to label each activation as either *stored* (at a boundary) or *recomputed*
(in the interior of a block).

Let $\mathbf{s} = [s_1, s_2, \dots, s_k]$ be the lengths of $k$ consecutive segments.
The cumulative sums
$$
b_i = \sum_{j=1}^{i-1} s_j,\quad i=1,\dots,k+1,
$$
give the indices of the boundaries (with $b_1 = 0$ and $b_{k+1}=L$).
Activations at indices $b_i$ are stored; all other indices are recomputed.

## Task

Implement `mark_activations`:

```python
def mark_activations(num_layers: int, seg_lengths: list[int]) -> list[int]:
    ...
```

* `num_layers`: total number of layers (positive integer).
* `seg_lengths`: a sequence of positive integers whose sum equals `num_layers`.
  Each element is the length of one segment.

The function must return a list of shape `(num_layers,)` and dtype `int`,
where entry `$i$` is `1` if activation $i$ lies on a checkpoint boundary
and `0` otherwise.  
If the sum of `seg_lengths` does not equal `num_layers`, raise `ValueError`.

The implementation should be fully vectorised; no explicit Python loops over layers.

## Example

```python

# 10 layers, split into segments of lengths [3, 4, 3]
labels = mark_activations(10, [3, 4, 3])
print(labels)
```

Output:

```
[1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
```

The stored activations are at indices `0`, `3`, and `7`.

## What the gate checks

* **Exact match**: The returned array must be identical to a reference
  computed by an oracle that follows the cumulative‑sum rule above.
* **Input validation**: A mismatch between `num_layers` and the sum of
  `seg_lengths` must raise `ValueError`.

The grader runs several random tests; any deviation from the exact
reference causes the gate to fail.
