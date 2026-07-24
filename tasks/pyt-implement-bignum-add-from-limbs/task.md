## Context

CPython stores arbitrary precision integers as arrays of fixed-width digits called
limbs. A simplified model uses base $2^{30}$, so each limb stores a value in the
range $0 \leq x < 2^{30}$.

A non-negative integer can be represented as

$$
N = \sum_{i=0}^{k-1} a_i (2^{30})^i ,
$$

where $a_i$ are the limbs in little-endian order. Adding two numbers requires
adding corresponding limbs and propagating carries:

$$
s_i = a_i + b_i + c_i .
$$

The output limb is

$$
r_i = s_i \bmod 2^{30},
$$

and the next carry is

$$
c_{i+1} = \left\lfloor \frac{s_i}{2^{30}} \right\rfloor .
$$

This models the low-level arithmetic performed by a big integer
implementation. The addition should be performed on the limb arrays directly.

## Task

Implement `add_limbs(a, b)`:

```python
def add_limbs(a: list[int], b: list[int]) -> list[int]:
    ...
```

The inputs are non-negative integers represented as little-endian lists of
30-bit limbs. Return a new little-endian limb list representing their sum.

The function must:

- add values using per-limb arithmetic and carry propagation;
- keep limbs in the range $0 \leq x < 2^{30}$;
- remove unnecessary high zero limbs, except that zero may be returned as `[0]`;
- not reconstruct the complete integers and use Python integer addition.

## Example

```python
a = [2**30 - 1, 5]
b = [1, 7]

print(add_limbs(a, b))
# [0, 6, 1]
```

The first limb overflows because

$$
(2^{30}-1)+1 = 2^{30},
$$

so it becomes zero and produces a carry into the next limb.

## What the gate checks

The gate uses a real Python integer oracle to convert limb arrays into values,
compute the sum, and convert the result back into limbs. The submitted function
must return exactly the same limb representation.

The gate also checks the implementation shape. Reconstructing the full integers
and applying a whole-number addition shortcut is rejected. The implementation
must use limb-level carry propagation.
