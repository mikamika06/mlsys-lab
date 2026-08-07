## Context

Causal self-attention restricts row $i$ to only attend to positions
$j \le i$. The correct way to enforce this is to set the disallowed
**logits** to $-\infty$ before the softmax:

$$
\text{score}_{i,j} =
\begin{cases}
\dfrac{q_i \cdot k_j}{\sqrt d} & j \le i \\[4pt]
-\infty & j > i
\end{cases}
\qquad
P_i = \operatorname{softmax}(\text{score}_{i,:})
$$

Because $\exp(-\infty) = 0$, this makes disallowed positions contribute
exactly $0$ probability while the *allowed* positions' softmax still sums
to $1$ — row $i$'s attention distribution is a valid probability
distribution over $\{0, \dots, i\}$.

A common bug instead computes the **unmasked** softmax first and then
zeroes out the disallowed probabilities afterward
($P_i \leftarrow P_i \odot \mathbb{1}[j \le i]$). This looks
superficially similar (the disallowed entries end up $0$ either way) but
the *allowed* entries were normalized against the wrong denominator — the
sum over **all** $n$ positions instead of just the $i+1$ allowed ones —
so row $i$'s masked probabilities now sum to less than $1$ and the output
is silently shrunk.

## Task

`tasks/rwa-debug-causal-mask-applied-after-softmax-instead-of-before/starter.py`
contains a broken `causal_self_attention` that masks the probabilities
after softmax instead of the logits before it. Fix it:

```python
def causal_self_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    ...
```

- `Q`, `K`, `V`: `(n, d)`.
- Row `i` of the output must be
  `softmax(scores[i, :i+1] / sqrt(d)) @ V[:i+1]`, i.e. computed by masking
  the logits with `-inf` for `j > i` **before** the softmax, not by
  zeroing probabilities after.
- Return shape `(n, d)`.

## Example

```python

Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]

out = causal_self_attention(Q, K, V)
# row 0 can only see position 0  -> out[0] == V[0]  == [10, 0]
# row 1 can see positions 0, 1   -> a softmax-weighted mix of V[0], V[1]
# row 2 can see all 3 positions  -> a softmax-weighted mix of all of V
```

## What the gate checks

The grader draws several random `(Q, K, V)` triples of varying `n`, `d`
from a seeded RNG and compares your output to a reference that masks the
logits with `-inf` before softmax, computed independently in Python —
never calling your function, never hardcoding an expected value.

`max_abs_err` is the worst per-case max-abs-error across all cases and
must be `<= 1e-5`. Masking probabilities after softmax instead of logits
before it leaves every row under-normalized by a factor that grows with
how many future positions were masked out, producing an error far above
this threshold on rows with `i < n - 1`.
