## Context

In many parallel algorithms a reduction operation (e.g. summation) can be executed in different orders.  
The *critical path* of a computation is the longest chain of dependent operations and determines its minimum possible latency on an ideal machine that can execute independent tasks simultaneously.

For a binary reduction over $N$ operands with an operation of constant latency $\ell$ we have two common execution patterns:

1. **Serial (chain) reduction**  
   The operands are combined one after the other:
   \[
   (((a_1 \;\texttt{op}\; a_2)\;\texttt{op}\; a_3)\;\dots\;\texttt{op}\; a_N).
   \]
   The critical‑path length is
   $$(N-1) \times \ell.$$

2. **Balanced binary-tree reduction**  
   At each level the number of active operands halves:
   \[
   (a_1 \;\texttt{op}\; a_2),\;
   (a_3 \;\texttt{op}\; a_4),\dots
   \]
   after which the results are combined again, and so on.  
   The depth of this tree is $\lceil \log_2 N\rceil$; therefore the critical‑path length equals
   \[
   \bigl\lceil \log_2 N \bigr\rceil \times \ell .
   \]

The task below asks you to compute these two lengths.

## Task

Implement the function `critical_path_lengths(n, latency)`:

```python
def critical_path_lengths(n: int, latency: int) -> tuple[int, int]:
    ...
```

* `n` – number of operands (positive integer).
* `latency` – cost of a single binary operation in cycle units (positive integer).

The function must return a tuple `(serial, tree)` where

* `serial` is the critical‑path length for the serial chain reduction.
* `tree`   is the critical‑path length for the balanced binary‑tree reduction.

Both values should be returned as plain integers.

## Example

```python
>>> from your_module import critical_path_lengths
>>> critical_path_lengths(8, 3)
(21, 9)       # (7*3, ceil(log2 8)*3 = 3*3)

>>> critical_path_lengths(7, 5)
(30, 15)      # (6*5, ceil(log2 7)*5 = 3*5)
```

## What the gate checks

The grader calls your implementation with a set of test cases and compares the result pair against the reference implementation. The comparison is an *exact match* on integer values; any deviation causes failure.
