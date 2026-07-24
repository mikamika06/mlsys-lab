## Context

In a parallel (tree) reduction, $N$ values are combined pairwise at each step. At every level of the reduction tree, adjacent pairs are merged, producing $\lceil N/2 \rceil$ values for the next level. The process repeats until a single value remains.

The number of levels (tree depth) satisfies the recurrence

$$d(1) = 0, \qquad d(N) = 1 + d\!\left(\left\lceil \frac{N}{2} \right\rceil\right) \text{ for } N > 1.$$

Solving this gives a closed-form depth of $\lceil \log_2 N \rceil$. For instance, $N = 5$ requires $\lceil \log_2 5 \rceil = 3$ steps: level 1 reduces $5 \to 3$, level 2 reduces $3 \to 2$, and level 3 reduces $2 \to 1$.

This depth bound underlies parallel prefix (scan) operations, parallel summation, and any associative reduction on $N$ elements with unlimited parallelism. It also appears as the height of a tournament bracket and the number of rounds in a butterfly network.

## Task

Implement `reduction_depth(N)`:

```python
def reduction_depth(N: int) -> int:
    ...
```

Given a positive integer $N \ge 1$, return the minimum number of parallel reduction steps required to combine $N$ elements into one using pairwise associative operations. The answer is $\lceil \log_2 N \rceil$.

You may use any approach: the closed-form `(N - 1).bit_length()`, a loop halving $N$ with ceiling division, or `math.ceil(math.log2(N))` (though beware floating-point edge cases for large $N$).

## Example

```python
reduction_depth(1)  # 0  — already a single element
reduction_depth(2)  # 1  — one pairwise combine
reduction_depth(4)  # 2  — 4 → 2 → 1
reduction_depth(5)  # 3  — 5 → 3 → 2 → 1
reduction_depth(8)  # 3  — 8 → 4 → 2 → 1
reduction_depth(10) # 4  — 10 → 5 → 3 → 2 → 1
```

## What the gate checks

The gate calls your function on a range of values: $N = 1$, powers of two ($2^k$), non-powers-of-two, and values above $2^{20}$. Each result is compared against the exact integer $\lceil \log_2 N \rceil$, computed independently by an oracle that iteratively applies $\lceil N/2 \rceil$ until reaching $1$. All results must match exactly — floating-point approximations or off-by-one errors on non-powers-of-two cause failure.
