## Context

In many transformer implementations sequences are concatenated into a single 2‑D tensor for efficient GPU kernels. The layout is described by an integer array `cu_seqlens` of length `S+1`, where `S` is the number of sequences. For each sequence `s` we have

$$
\text{start}_s = \text{cu\_seqlens}[s], \qquad
\text{end}_s   = \text{cu\_seqlens}[s+1],
$$

and the slice `packed[start_s:end_s]` contains all rows belonging to that sequence. Reconstructing the original per‑sequence tensors is a common preprocessing step before applying sequence‑specific operations.

## Task

Implement `unpack_sequences(packed, cu_seqlens)`:

```python
def unpack_sequences(packed, cu_seqlens):
    ...
```

`packed` has shape `(N, D)` where `N = \sum_i L_i` and each row is a token embedding. `cu_seqlens` is a 1‑D integer array of length `S+1`. The function must return a list of `S` list, each with shape `(L_i, D)`, corresponding to the original sequences.

The implementation should use only Python slicing; no explicit Python loops over tokens. It may use a loop over the number of sequences (which is typically small), but it must not iterate over individual rows.

## Example

```python

packed = [[0, 1],
                   [2, 3],
                   [4, 5],
                   [6, 7]]
cu_seqlens = [0, 2, 4]   # two sequences: first length 2, second length 2

seqs = unpack_sequences(packed, cu_seqlens)
# seqs[0] == array([[0,1],[2,3]])
# seqs[1] == array([[4,5],[6,7]])
```

## What the gate checks

The grader computes a reference list of slices using `cu_seqlens` and compares it element‑wise with the student's output via `==`. The metric `exact_match` must be `== 1.0`.
