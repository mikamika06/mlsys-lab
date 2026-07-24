## Context

Python computes method resolution order (MRO) using C3 linearization. For a class $C$ with direct bases $B_1, B_2, \dots, B_n$, C3 computes

$$
\mathrm{MRO}(C) = [C] + \mathrm{merge}(\mathrm{MRO}(B_1), \mathrm{MRO}(B_2), \dots, \mathrm{MRO}(B_n), [B_1, B_2, \dots, B_n]).
$$

The merge operation repeatedly chooses a head that is not present in the tail of any other list. Such a head is called a good head. The chosen heads form the linearized order after the class itself.

For example, if a class has bases that produce a merge result beginning with classes $A, B, O$, the good heads selected by the merge are $A$, then $B$, then $O$.

## Task

Implement `c3_merge_trace(bases)`:

```python
def c3_merge_trace(bases):
    ...
```

`bases` is a tuple of Python classes. The function must return a list containing the class names in the order that C3 selects good heads while merging the base MRO lists.

Do not return the name of the class being created. Return only the merge selections after the class itself. The result must include every class selected by the merge, including `object` when it appears.

## Example

```python
class O:
    pass

class A(O):
    pass

class B(O):
    pass

trace = c3_merge_trace((A, B))
# ["A", "B", "O", "object"]
```

## What the gate checks

The gate creates real Python class hierarchies and uses CPython's computed `mro()` as the oracle for the expected C3 linearization. Your implementation is checked against the ordered class names after the newly created class. The returned sequence must exactly match the CPython MRO result.
