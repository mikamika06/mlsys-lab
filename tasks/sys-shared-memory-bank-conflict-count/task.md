## Context

GPU shared memory is divided into banks that can serve independent addresses in parallel. In a simplified 32-bank model, a warp has 32 lanes and lane $i$ accesses a shared-memory word address $a_i$.

The bank for an address is

$$
b_i = a_i \bmod 32 .
$$

If multiple lanes in the same warp access the same bank, the accesses are serialized. The conflict degree for a warp is the maximum number of lanes targeting any single bank:

$$
c = \max_{k \in \{0,\dots,31\}} \left|\{i : b_i = k\}\right| .
$$

A conflict degree of $1$ means every lane accesses a different bank. A value larger than $1$ indicates bank conflicts.

## Task

Implement `bank_conflict_degree(accesses)`:

```python
def bank_conflict_degree(accesses):
    ...
```

The argument `accesses` is a list of warps. Each warp is a list of exactly 32 non-negative integer shared-memory word addresses. Return a list containing the conflict degree for each warp.

The function should model a 32-bank shared-memory system. Do not assume that addresses are already converted to banks; the function must perform the modulo operation.

## Example

```python
accesses = [
    list(range(32)),
    [0] * 32,
]

result = bank_conflict_degree(accesses)
# [1, 32]
```

The first warp maps each lane to a different bank. The second warp maps every lane to bank $0$.

## What the gate checks

The gate builds several access patterns and compares the returned conflict degrees against a reference implementation of the 32-bank model. The `modeled_mem_access` score is `1.0` only when all warp conflict degrees match the computed reference.
