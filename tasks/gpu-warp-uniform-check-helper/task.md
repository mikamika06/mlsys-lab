## Context

In SIMT GPUs a *warp* is the group of threads that execute the same instruction in lock‑step.  
If every thread in a warp has the same predicate (branch condition), the execution can be
predicated safely; otherwise divergence forces serialisation of the divergent paths.
A common optimisation pattern is to check whether all lanes in a warp agree on a value
before deciding to take or skip an expensive branch.

The helper function we want implements this test.  
Its input represents the per‑lane predicates (typically booleans or 0/1 integers).  
It should return ``True`` only when *all* 32 lanes share the same value.

## Task

Implement a Python function with the following signature:

```python
def all_lanes_agree(pred):
    """
    Return True iff every element in pred has the same truth‑value.
    The length of pred is assumed to be 32 (a warp), but the implementation may
    handle arbitrary lengths gracefully. No mutation of pred is required.
    """
```

The function must not depend on any external libraries beyond the standard library and Python.

## Example

```python
>>> all_lanes_agree([True] * 32)
True
>>> all_lanes_agree([True, True, False] + [True]*29)
False
>>> all_lanes_agree([0] * 32)   # integers are accepted
True
```

## What the gate checks

The grading script evaluates your implementation against a handful of deterministic test cases
and verifies that every return value exactly matches the reference solution.  
Only an exact match on the boolean outputs is required; there is no performance or memory constraint.
