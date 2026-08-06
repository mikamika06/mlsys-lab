## Context

The cross‑entropy loss for a single example with logits $z \in \mathbb{R}^C$ and target class $y$ is

$$\ell(z, y) = -\,\log p_y,\qquad
p_j = \frac{\exp(z_j)}{\sum_{k=1}^{C}\exp(z_k)}.$$

For a batch of $N$ examples the average loss is

$$L = \frac{1}{N}\sum_{i=1}^N \ell(z^{(i)}, y^{(i)}).$$

In many sequence‑modeling tasks some positions are padding and should not influence training.  The convention in PyTorch and Python is to use an *ignore index* $I$; any target equal to $I$ must be omitted from both the numerator and denominator of the average:

$$L_{\text{masked}} = \frac{1}{M}\sum_{i:\,y^{(i)}\neq I} \ell(z^{(i)}, y^{(i)}),$$

where $M$ is the number of non‑ignored positions.

## Task

Implement a function that computes this masked average cross‑entropy loss:

```python
def masked_cross_entropy(logits: list[list[float]],
                         targets: list[int],
                         ignore_index: int = -100) -> float:
    ...
```

* `logits` is an $(N, C)$ array of raw scores.
* `targets` is a length‑$N$ integer array with values in $\{0,\dots,C-1\}$ or the special value `ignore_index`.
* The function must return a scalar `float64` representing $L_{\text{masked}}$.
* If all positions are ignored, return `0.0`.

The implementation should use only vectorised Python operations; no explicit Python loops.

## Example

```python

logits = [[2.0, 1.0],
                   [0.5, 1.5],
                   [1.0, 3.0]]
targets = [0, -100, 1]   # second position is padding

loss = masked_cross_entropy(logits, targets)
print(loss)  # ≈ 0.6931471805599453
```

The loss is the average of the two non‑ignored examples.

## What the gate checks

The grader computes a reference implementation using Python’s log‑softmax and masking.  
It then compares your result to that reference with the metric `max_abs_err`.  
Your solution must satisfy

```
max_abs_err ≤ 1e-6
```

to pass the gate.
