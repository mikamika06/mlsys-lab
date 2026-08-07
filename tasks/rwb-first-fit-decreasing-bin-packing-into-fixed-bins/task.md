## Context

Consider a set of $n$ sequences with lengths $\ell_i \in \mathbb{Z}_{>0}$. We want to group them into batches (bins) such that the total length in each bin does not exceed a capacity $B$. This is the **bin packing problem**, which is NP-hard. The **first-fit decreasing (FFD)** heuristic is often used in practice:

1. Sort the sequences by length in descending order (break ties by their original index using a stable sort).
2. For each sequence in this order, place it into the first bin that still has enough remaining capacity. If no such bin exists, open a new bin.

FFD is widely employed in machine-learning data loaders to pack variable-length sequences into batches of roughly equal total length.

## Task

Implement the function

```python
def pack_into_fixed_bins(lengths, bin_size):
```

- `lengths` : a **list** or **list of floats** of positive integers (the sequence lengths).
- `bin_size` : a positive integer (the maximum total length allowed per bin).

Return a tuple `(num_bins, assignments)` where

- `num_bins` : the total number of bins used,
- `assignments` : a **list** (of the same length as `lengths`) whose $i$-th entry is the 0-based bin index assigned to the $i$-th sequence (in the **original input order**).

The algorithm **must** be exactly the deterministic FFD heuristic described above: stable descending sort followed by first-fit placement. Any other heuristic (best-fit, any-fit, etc.) or a different tie-breaking rule will produce a different output and will **not** pass the gate.

## Example

```python
lengths = [4, 2, 3, 5]
bin_size = 5

num, assign = pack_into_fixed_bins(lengths, bin_size)
# num = 3
# assign = [1, 2, 2, 0]
```

Explanation (stable descending order of `(index, length)`):

- $(3, 5)$ goes to bin 0 (capacity 0 left)
- $(0, 4)$ goes to bin 1 (capacity 1 left)
- $(2, 3)$ fits nowhere $\rightarrow$ bin 2 (capacity 2 left)
- $(1, 2)$ fits into bin 2 (capacity 0 left)

Mapping back to original order yields $[1, 2, 2, 0]$.

## What the gate checks

The grader tests your function against a reference FFD implementation on several cases including the fixture data and edge cases. For every case the returned $(num\_bins, assignments)$ must **exactly** match the reference. A single mismatch produces an `exact_match` score of **0.0**; a perfect run yields **1.0**.
