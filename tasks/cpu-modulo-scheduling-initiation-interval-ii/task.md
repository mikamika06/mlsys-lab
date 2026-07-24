## Context

Modulo scheduling is a technique used in software pipelining to optimize loops for parallel execution. The initiation interval (II) is a key parameter that determines how frequently iterations of a loop can be initiated. It is defined as the maximum of the resource-bound and the recurrence-bound:

- The **resource-bound** is determined by the available resources (e.g., functional units) and the demands of the loop body.
- The **recurrence-bound** is determined by the data dependencies within the loop.

The initiation interval is crucial for achieving optimal performance in loop unrolling and software pipelining.

## Task

Implement the function `compute_initiation_interval(resource_bound, recurrence_bound)`:

```python
def compute_initiation_interval(resource_bound: int, recurrence_bound: int) -> int:
    ...
```

This function takes two integers, `resource_bound` and `recurrence_bound`, and returns the initiation interval (II), which is the maximum of the two bounds.

## Example

```python
ii = compute_initiation_interval(3, 5)
# ii should be 5

ii = compute_initiation_interval(6, 4)
# ii should be 6
```

## What the gate checks

The gate checks for an exact match between the computed initiation interval and the reference solution. The solution must correctly compute the initiation interval as the maximum of the resource-bound and recurrence-bound. The grading is deterministic and relies on a cache simulation to ensure the correct access pattern and value computation.
