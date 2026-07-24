## Context

Scaled Dot‑Product Attention (SDPA) is a core component of transformer models.  
Given query, key and value matrices $Q,K,V \in \mathbb{R}^{S\times d}$, the attention output is

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

The dominant cost comes from two matrix multiplications:

1. $QK^\top$ – a $(S\times d)\times(d\times S)$ product, producing an $S\times S$ attention score matrix.
2. The weighted sum $AV$, where $A$ is the softmaxed scores and $V$ is $(S\times d)$.  
   This is again an $S\times S$ times $(S\times d)$ multiplication.

For a single multiply–add pair we count two floating‑point operations (one multiplication, one addition).  Therefore

$$\operatorname{FLOPs}(QK^\top)=2\,S^2d,$$
$$\operatorname{FLOPs}(AV)=2\,S^2d.$$

The total number of FLOPs for SDPA is thus

$$f(S,d)=4\,S^2d.$$

## Task

Implement the function `sdpa_flop_closed_form` that, given the sequence length $S$ and feature dimension $d$, returns the exact number of floating‑point operations required to compute scaled dot‑product attention as described above.

```python
def sdpa_flop_closed_form(S: int, d: int) -> int:
    ...
```

The function must return an integer and use only pure Python arithmetic; no external libraries are needed.

## Example

```python
>>> sdpa_flop_closed_form(3, 4)
144
# Explanation: 4 * 3^2 * 4 = 144
```

## What the gate checks

The grader evaluates your implementation on several $(S,d)$ pairs and compares the returned value to the reference computed by the closed‑form expression $4\,S^2d$.  The comparison uses exact integer equality (`==`).  Any deviation causes a failure.
