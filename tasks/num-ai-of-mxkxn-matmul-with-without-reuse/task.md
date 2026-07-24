## Context

The roofline model relates computational performance to memory bandwidth via arithmetic intensity  
$Q = F / M$, where $F$ is the number of floating‑point operations (flops) and $M$ is the number of
bytes transferred to/from main memory.

For a matrix multiplication $\mathbf{C}_{M \times N} = \mathbf{A}_{M \times K} \cdot \mathbf{B}_{K \times N}$, the
flop count is

$$F = 2 M N K.$$

(The factor 2 comes from counting a fused multiply–add as one operation; the underlying adds and multiplies
are each one flop.)

The memory traffic depends on how effectively data is reused.

* **No reuse** (naïve unblocked): every multiply–add loads an element of $\mathbf{A}$ and an element of
$\mathbf{B}$ from memory.  Additionally the result $\mathbf{C}$ is written once.  Hence

$$M_{\text{no reuse}} = 8\,(2 M N K + M N).$$

* **Full cache reuse** (perfect tiling): each element of $\mathbf{A}$ and $\mathbf{B}$ is loaded exactly once
and $\mathbf{C}$ is written once:

$$M_{\text{full reuse}} = 8\,(M K + K N + M N).$$

The arithmetic intensities are therefore

$$Q_{\text{no reuse}} = \frac{F}{M_{\text{no reuse}}}, \qquad
Q_{\text{full reuse}} = \frac{F}{M_{\text{full reuse}}}.$$

## Task

Implement the function  

```python
def ai_matmul(M: int, K: int, N: int) -> tuple[float, float]:
```

that returns the arithmetic intensity for the two scenarios: `(AI_no_reuse, AI_full_reuse)`.
Use the formulas above with double‑precision elements (8 bytes per element).  
The function must work for any positive integers $M, K, N$.

## Example

```python
>>> ai_matmul(64, 64, 64)
(0.124, 5.333)   # approximate values
```

(The exact numbers are $0.124\ldots$ and $5.333\ldots$; the gate checks the relative error.)

## What the gate checks

The grader evaluates your function for several $(M, K, N)$ triples. It compares the returned pair against
the reference using the relative $L_2$ error over all values:

$$\text{rel\_err} = \frac{\lVert\mathbf{v}_{\text{student}} - \mathbf{v}_{\text{ref}}\rVert}
{\lVert\mathbf{v}_{\text{ref}}\rVert}.$$

The gate passes when $\text{rel\_err} \le 10^{-6}$.
