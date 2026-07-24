## Context

Iterators in Python are evaluated on demand. A generator can represent a pipeline stage without computing all outputs immediately.

Suppose a producer maps inputs $0,1,\dots,N-1$ to their squares. A lazy consumer that requests only $K$ values should cause the producer to evaluate only those $K$ inputs:

$$
0^2, 1^2, \dots, (K-1)^2 .
$$

An eager implementation computes all $N$ outputs before the consumer starts. This wastes work when $K \ll N$.

The iterator protocol allows a generator function to pause after each `yield` and resume when the next item is requested.

## Task

Implement `take_k_squares(n, k)`:

```python
def take_k_squares(n: int, k: int):
    ...
```

The function must return an iterator that yields exactly the first $k$ squared values from the sequence of integers `0` through `n - 1`.

Requirements:

- The returned object must be lazy. Creating it must not compute all values.
- The consumer must be able to stop after receiving $k$ values without the producer processing the remaining $n-k$ inputs.
- Use the iterator protocol through a generator or an equivalent lazy iterator implementation.
- The yielded values must equal $[0^2, 1^2, \dots, (k-1)^2]$ for valid inputs with $0 \le k \le n$.

## Example

```python
it = take_k_squares(1000000, 3)

print(next(it))
print(next(it))
print(next(it))

# Output:
# 0
# 1
# 4
```

The iterator should not square the remaining $999997$ values.

## What the gate checks

The first gate consumes exactly $k$ values and compares them with a reference implementation that computes the same mathematical sequence.

The second gate traces Python line events inside the producer iterator. The reference generator establishes the expected producer work for $k$ yielded items. An eager implementation that computes all $n$ values before returning fails because its producer is not stopped after $k$ items.
