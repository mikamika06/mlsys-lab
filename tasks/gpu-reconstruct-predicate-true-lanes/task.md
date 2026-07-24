## Context

In a single‑warp GPU kernel each thread can test whether it should perform some work by evaluating a predicate `cond`. The CUDA warp primitives return an integer called a *ballot* where bit *i* is set if the predicate was true in lane *i*. For many algorithms we need to know which lanes actually satisfied the condition. This task asks you to reconstruct that list of lanes from a 32‑bit ballot mask.

A ballot mask `m` has its least significant bit corresponding to lane 0, next bit to lane 1 and so on up to lane 31. Thus, if a warp consists of lanes 0–31 and the predicate is true in lanes 2, 5, 27, the integer representation would be

$$
m = 2^2 + 2^5 + 2^{27} .
$$

## Task

Implement `reconstruct_lanes(mask)`:

```python
def reconstruct_lanes(mask: int) -> List[int]:
    ...
```

The function receives a 32‑bit unsigned integer and returns a Python list containing all lane indices (0 ≤ i < 32) whose bit is set in *mask*. The list must be sorted in ascending order. Do not use any external libraries beyond the standard library.

## Example

```python
>>> reconstruct_lanes(0b10101)
[0, 2, 4]
```

Here lane 0, 2 and 4 had their predicate true; all other lanes were false.

## What the gate checks

The grader compares your return value with a reference implementation using exact equality. The output must be a Python list of integers sorted from smallest to largest. Any deviation – wrong values, wrong order or incorrect type – will cause the gate to fail.
