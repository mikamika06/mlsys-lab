## Context

In many machine learning pipelines we compute the softmax of a large batch of logits in blocks. For each block we often keep three running statistics that allow us to merge the results without recomputing from scratch:

- $m$ – the maximum logit value in the block,
- $s$ – the sum $\displaystyle \sum_i e^{\,\ell_i - m}$ of exponentials shifted by $m$,
- $w$ – the weighted sum $\displaystyle \sum_i \ell_i\,e^{\,\ell_i - m}$.

These three numbers are enough to recover the normalised softmax probabilities and their log‑likelihoods. The **online softmax lemma** states that if we have two blocks with statistics $(m_1,s_1,w_1)$ and $(m_2,s_2,w_2)$, then the combined block has

$$
\begin{aligned}
m &= \max(m_1,m_2),\\[4pt]
s &= s_1\,e^{\,m_1-m} + s_2\,e^{\,m_2-m},\\[4pt]
w &= w_1\,e^{\,m_1-m} + w_2\,e^{\,m_2-m}.
\end{aligned}
$$

The exponentials are always taken with a negative shift, so the computation is numerically stable even when logits differ by several orders of magnitude.

## Task

Implement `combine_softmax_stats(block_a, block_b)` that takes two tuples `(m,s,w)` and returns the combined statistics as a tuple `(m,s,w)`. The function must use only Python for any arithmetic; no explicit Python loops are required. All inputs and outputs should be plain Python floats (not Python scalars).

## Example

```python

# block 1: logits [0, 2]
m1 = 2.0
s1 = math.exp(0-2) + math.exp(2-2) # e^{-2} + 1
w1 = 0*math.exp(-2) + 2*1 # 2

# block 2: logits [5, -1]
m2 = 5.0
s2 = math.exp(5-5) + math.exp(-1-5) # 1 + e^{-6}
w2 = 5*1 + (-1)*math.exp(-6)

combined = combine_softmax_stats((m1,s1,w1), (m2,s2,w2))
print(combined)
# Expected: (5.0, s_combined, w_combined) where
# s_combined = s1*e^{2-5} + s2*e^{5-5}
# w_combined = w1*e^{2-5} + w2*e^{5-5}
```

## What the gate checks

The grader computes a reference implementation using Python and compares your result with it. The maximum absolute elementwise error must be at most $10^{-9}$.
