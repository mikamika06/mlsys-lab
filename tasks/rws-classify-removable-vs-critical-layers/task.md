## Context

In many neural‑network compression pipelines a *Batch Importance* (BI) score is assigned to each layer. A small BI indicates that the activations of that layer change little when the network is trained on a fresh batch, and such layers are often considered **removable**: they can be pruned or fused without harming accuracy. Conversely, a large BI signals that the layer captures essential information and must be kept.

Mathematically, given an array $\mathbf{b} \in \mathbb{R}^{L}$ of BI values for $L$ layers and a scalar threshold $\tau$, we classify layer $i$ as removable iff
$$
b_i < \tau .
$$

The task is to implement this simple decision rule.

## Task

Implement `classify_removable_layers(bis, threshold)`:

```python
def classify_removable_layers(bis: list[float], threshold: float) -> Set[int]:
    ...
```

It receives a list of floats of BI scores and a floating point threshold. It must return the set of indices (Python `int`) that satisfy $b_i < \tau$.

The function should be pure; no side effects or global state.

## Example

```python
bis = [0.12, 0.85, 0.47, 0.63]
threshold = 0.6
removable = classify_removable_layers(bis, threshold)
# removable == {0, 2}
```

## What the gate checks

The grader computes a reference set using Python’s vectorised comparison and compares it with the learner’s output via exact set equality. The metric is `exact_match`; the solution must return exactly the same indices as the oracle.
