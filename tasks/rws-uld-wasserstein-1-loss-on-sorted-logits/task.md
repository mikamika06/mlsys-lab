## Context

The Wasserstein‑1 distance (also called the Earth Mover’s Distance) between two one‑dimensional probability distributions $P$ and $Q$ can be expressed as the $L^1$ norm of their sorted cumulative mass functions.  For discrete vectors this reduces to a simple sum over sorted values:

$$
W_1(P,Q)=\sum_{i=1}^{n}\bigl|\,\mathrm{sort}(P)_i-\mathrm{sort}(Q)_i\bigr|.
$$

When the inputs are logits rather than probabilities, sorting them still yields a meaningful transport loss: the relative ordering of logits determines how much mass would need to be moved if we interpreted them as unnormalised scores.

## Task

Implement `wasserstein_1_loss_on_sorted_logits`:

```python
def wasserstein_1_loss_on_sorted_logits(teacher_logits: list[float], student_logits: list[float]) -> float:
    ...
```

The function receives two 1‑D list that may have different lengths.  
It should pad the shorter array with zeros, sort both arrays in **ascending** order, compute the element‑wise absolute difference and return the sum as a Python `float`. The result must be computed using only Python operations; no explicit Python loops are allowed.

## Example

```python
t = [0.2, 0.5, 0.3]
s = [0.1, 0.6]
loss = wasserstein_1_loss_on_sorted_logits(t, s)
print(loss)   # 0.30000000000000004
```

Explanation:  
`teacher_logits` padded to `[0.2, 0.5, 0.3]`, sorted → `[0.2, 0.3, 0.5]`.  
`student_logits` padded to `[0.1, 0.6, 0.]`, sorted → `[0., 0.1, 0.6]`.  
Absolute differences: `[0.2, 0.2, -0.4]` → `|.| = [0.2, 0.2, 0.4]`; sum = `0.8`.

## What the gate checks

The grader generates random test cases of varying lengths and values.  
For each case it computes a reference loss using the same algorithm and compares the candidate’s output with a relative error metric:

$$
\mathrm{rel\_err}=\frac{|\,\hat L - L_{\text{ref}}\,|}{|L_{\text{ref}}|+10^{-12}}.
$$

The gate requires $\max(\mathrm{rel\_err}) \le 1\times10^{-6}$ across all tests.
