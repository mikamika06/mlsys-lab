## Context

Python resolves multiple inheritance with C3 linearization. The algorithm combines local
class precedence orders and parent linearizations while preserving monotonicity.

For a class $C$ with bases $B_1, B_2, \dots, B_n$, C3 computes an order by repeatedly
selecting a valid head from the candidate sequences:

$$
L[C] = [C] + merge(L[B_1], L[B_2], \dots, L[B_n], [B_1, B_2, \dots, B_n]).
$$

A candidate head can be selected only when it does not appear in the tail of any other
sequence. This rule prevents inconsistent ordering when several inheritance paths meet.

A diamond hierarchy is a common example:

$$
\begin{array}{c}
\quad A \\
/ \quad \backslash \\
B \quad C \\
\backslash \quad / \\
\quad D
\end{array}
$$

Python computes the method resolution order (MRO) for `D` as a sequence that starts with
`D` and ends with `object`.

## Task

Implement `diamond_merge(cls)`:

```python
def diamond_merge(cls):
    ...
```

The function receives a Python class object and returns a list of class names representing
the C3 method resolution order of that class. The returned list must include the input
class and all classes in its MRO, including `"object"` when present.

Do not use `cls.__mro__` or `type.mro()` directly. The goal is to practice deriving the
C3 merge result.

## Example

```python
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass

diamond_merge(D)
# ["D", "B", "C", "A", "object"]
```

## What the gate checks

The gate builds several inheritance hierarchies and compares the returned name sequence
with the order produced by the CPython interpreter's real MRO implementation.

The `exact_match` score is $1.0$ only when every tested hierarchy has the exact same
sequence of names.
