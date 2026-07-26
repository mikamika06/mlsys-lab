## Context

In a transformer language model, the final linear layer (the *LM head*) maps each hidden state vector $h_i \in \mathbb{R}^d$ to logits over the vocabulary of size $V$.  
With weight tying, the same matrix $W \in \mathbb{R}^{V\times d}$ is used for both embedding and output projection.  
A forward pass for a sequence of length $S$ therefore requires computing

$$
\text{logits} = H\,W^{\top},
$$

where $H \in \mathbb{R}^{S\times d}$ contains the hidden states.  
Each element of the resulting matrix $\mathbb{R}^{S\times V}$ is a dot product of two length‑$d$ vectors, which costs $2d$ floating point operations (one multiplication and one addition per dimension).  Hence

$$
\text{FLOPs} = 2 \times S \times d \times V .
$$

This simple closed form captures the arithmetic intensity of the LM head.

## Task

Implement a function `lm_head_flops(S: int, d: int, V: int) -> int` that returns the exact number of floating point operations required for one forward pass through the LM head given:

* $S$ – sequence length (number of tokens),
* $d$ – hidden dimension,
* $V$ – vocabulary size.

The function should use only pure Python arithmetic and return an integer.  No loops or external libraries are necessary.

## Example

```python
>>> lm_head_flops(1, 768, 50257)
77194752
```

Explanation:  
$2 \times 1 \times 768 \times 50257 = 77\,194\,752$ FLOPs.

## What the gate checks

The grader evaluates your implementation on several random configurations and verifies that the returned value equals the reference formula $2S d V$.  The metric is `exact_match`; any deviation causes failure.  Because the computation is constant‑time, no additional performance constraints are imposed.
