## Context

In transformer inference, each decoding step requires the weights of every layer. When the model is off‑loaded to a device with limited memory, an **async prefetch/evict** scheduler keeps only a small resident set in RAM while the rest are streamed from disk or network.  
At any time the resident set must contain the *current* layer and the *next* layer so that the next step can start immediately after the current one finishes. After a layer has been used it is evicted, and the following layer is prefetched.

The scheduler therefore maintains a sliding window of size two over the circular list of layers $\{0,\dots,L-1\}$.

## Task

Implement `emit_resident_set_per_step(num_layers: int, num_steps: int) -> List[Set[int]]`:

```python
def emit_resident_set_per_step(num_layers: int, num_steps: int):
    ...
```

The function should return a list of length `num_steps`.  
Each element is the set of layer indices that are resident in memory **at the start** of that decoding step. The algorithm must follow the rules:

1. At step 0 the resident set contains layers `0` and, if it exists, `1`.
2. For every subsequent step `s>0`:
   * Evict the current layer `(s-1) mod L`.
   * Prefetch the layer two steps ahead `((s+1)+1) mod L = (s+2) mod L`.

The resident set must never contain more than two layers.

## Example

```python
>>> emit_resident_set_per_step(3, 5)
[{0, 1}, {1, 2}, {2, 0}, {0, 1}, {1, 2}]
```

Here the list shows the resident set at steps 0–4 for a model with three layers.

## What the gate checks

Two metrics are evaluated:

* **exact_match** – the candidate’s output must be *identical* to the reference implementation for all test cases.
* **peak_resident** – the maximum size of any resident set across all steps must not exceed 2.  

Both metrics use strict equality or inequality; any deviation causes the gate to fail.
