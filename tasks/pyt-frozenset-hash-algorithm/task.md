## Context

A `frozenset` is an immutable set whose hash must not depend on the order in
which elements were inserted. CPython computes the hash by combining the hashes
of the members with an order-independent mixing function.

The core combination uses XOR:

$$h = m_1 \oplus m_2 \oplus \dots \oplus m_n$$

where each member hash is transformed by a bit mixing function before being
combined. Since XOR is commutative, the result is independent of iteration
order.

After combining member hashes, CPython mixes in the number of elements and
performs additional scrambling with fixed-width unsigned integer arithmetic.
The final bit pattern is interpreted as a signed Python hash value.

## Task

Implement `frozenset_hash(values)`:

```python
def frozenset_hash(values):
    ...
```

The input is an iterable of integers. Return the same integer produced by
CPython for:

```python
hash(frozenset(values))
```

Do not call `hash(frozenset(values))` directly. Implement the hashing procedure
using integer arithmetic and the member hashes.

The implementation only needs to support integer members. Integer hash
randomization does not affect these values in the target interpreter.

## Example

```python
print(frozenset_hash([1, 2, 3]))
# matches hash(frozenset([1, 2, 3]))

print(frozenset_hash([3, 2, 1]))
# returns the same value
```

## What the gate checks

The gate computes the CPython frozenset hash algorithm independently and compares
the submitted implementation against that reference on several integer
collections.

The metric is `exact_match`. The implementation must return exactly the same
integer as the reference for every checked case.
