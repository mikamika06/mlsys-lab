## Context

CPython sets use open addressing to find candidate slots in a hash table. A set lookup starts at the slot determined by the hash value:

$$i = \mathrm{hash}(key) \mathbin{\&} mask$$

The probing strategy has two phases. It first checks a small linear run of slots. For each linear probe it advances by one position:

$$i = (i + 1) \mathbin{\&} mask$$

After the linear probes are exhausted, CPython uses a perturbation sequence to spread later probes across the table:

$$i = (5i + 1 + perturb) \mathbin{\&} mask$$

where the perturb value is repeatedly shifted:

$$perturb = perturb \gg 5$$

The mask is one less than the table size, so the table size is expected to be a power of two.

## Task

Implement `set_probe_sequence(key, mask)`:

```python
def set_probe_sequence(key: int, mask: int) -> list[int]:
    ...
```

Return the sequence of table slot indices visited by CPython's set probing algorithm for one key. Return the first 20 visited slots.

Use the integer key value as its hash value. The first slot must be `key & mask`. The sequence must include the linear probe phase followed by the perturbation phase. Do not use a Python `set` internally.

## Example

```python
print(set_probe_sequence(5, 7))
# [5, 6, 7, 0, 1, 2, 3, 4, ...]
```

The exact continuation depends on the perturbation updates after the linear probe phase.

## What the gate checks

The gate compares the returned sequences with a reference computed from CPython's set probing rules using integer hashes from Python's runtime. The comparison is exact: every slot index and ordering must match for all tested keys and table masks.
