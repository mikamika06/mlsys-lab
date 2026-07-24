## Context

Python dictionaries use hash tables with open addressing. Before probing for an empty entry, a key's first candidate position is determined from its hash value and the table mask.

For a power-of-two table size $m$, the initial slot is:

$$
s = \operatorname{hash}(k) \mathbin{\&} (m - 1).
$$

Different keys may map to the same initial slot even when they are distinct. Measuring this distribution helps analyze collision pressure before the later probe sequence begins.

Given integer keys $k_0, k_1, \dots, k_{n-1}$ and a table size $m$, an initial-slot histogram $h$ contains one count for each possible initial slot:

$$
h_i = |\{k_j : \operatorname{hash}(k_j) \mathbin{\&} (m-1) = i\}|.
$$

The histogram describes only the first hash-table position. It does not perform insertion, probing, or collision resolution.

## Task

Implement `initial_slot_histogram(keys, size)`:

```python
def initial_slot_histogram(keys, size):
    ...
```

The function receives a sequence of integer keys and a positive power-of-two table size. Return a list of length `size`, where each element is the number of keys whose initial slot equals that index.

Use Python's integer `hash` behavior. Do not simulate the full dictionary insertion algorithm.

## Example

```python
keys = [1, 9, 17, 2]
hist = initial_slot_histogram(keys, 8)

# hash(k) & 7 gives slots:
# 1 -> 1, 9 -> 1, 17 -> 1, 2 -> 2
# result:
# [0, 3, 1, 0, 0, 0, 0, 0]
```

## What the gate checks

The gate builds several key sets and table sizes and computes the expected histogram with the CPython integer `hash` implementation available in the running interpreter.

The returned list must exactly match the oracle histogram. The `exact_match` score must equal $1.0$.
