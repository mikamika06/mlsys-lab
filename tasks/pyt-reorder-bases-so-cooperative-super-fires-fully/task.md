## Context

Python determines method lookup order with C3 linearization. The method resolution
order (MRO) is a sequence of classes that defines how attribute lookup and
cooperative `super()` calls continue.

For a class $C$, Python computes an ordered list

$$
\mathrm{MRO}(C) = [C, B_1, B_2, \dots, \mathrm{object}].
$$

When classes use cooperative methods, each implementation calls `super()` and
expects the next class in the MRO to continue the chain. A bad direct base order
can either make class creation invalid or prevent the intended cooperative chain
from running.

## Task

Implement `reorder_bases(bases)`:

```python
def reorder_bases(bases):
    ...
```

`bases` is a list of classes that should become direct bases of a new combined
class. Return a tuple containing the same classes in an order that Python accepts
for multiple inheritance and that preserves the cooperative `super()` chain.

The returned value will be used as:

```python
class Combined(*reorder_bases(bases)):
    def run(self):
        return super().run()
```

The result of `Combined().run()` must include every cooperative base contribution.

## Example

```python
class Root:
    def run(self):
        return ["Root"]

class A(Root):
    def run(self):
        return ["A"] + super().run()

class B(A):
    def run(self):
        return ["B"] + super().run()

bases = [A, B]

ordered = reorder_bases(bases)

class Combined(*ordered):
    def run(self):
        return super().run()

# The repaired order is (B, A), so the chain is complete.
assert Combined().run() == ["B", "A", "Root"]
```

## What the gate checks

The gate uses Python's own class construction and `super()` behavior as the
oracle. It tries valid permutations of the provided bases, selects the ordering
that produces the complete cooperative method chain, and compares the submitted
ordering and result against that oracle.

The `exact_match` metric must be $1.0$. Returning the input order fails when the
input order conflicts with the C3 MRO constraints.
