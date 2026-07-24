## Context

Training a transformer requires storing intermediate activations from the
forward pass for use in the backward pass. Korthikanti et al. ("Reducing
Activation Recomputation in Large Transformer Models") derive a closed-form
byte count for the activations stored per transformer layer, without any
activation recomputation, under standard assumptions (activations kept in
16-bit precision, dropout masks stored as 1 byte per element):

$$
M(b, s, h, a) = s\,b\,h\left(34 + \frac{5 a s}{h}\right) \;\; \text{bytes},
$$

where:

- $b$ — micro-batch size,
- $s$ — sequence length,
- $h$ — hidden size,
- $a$ — number of attention heads.

The constant $34sbh$ term comes from the attention block's QKV/output
projections and the two-layer MLP (each stored in 16-bit precision) plus
their dropout masks and layer norms. The $5as^2b$ term (the $5as/h$ piece,
after multiplying through by $sbh$) comes from the attention probability
matrix and its dropout mask, which scale quadratically in sequence length
and linearly in the number of heads, independent of the hidden size.

## Task

Implement `activation_memory_bytes(b, s, h, a)`.

All four arguments are positive integers. Return the per-layer activation
memory in bytes, as a Python `float` (or anything that compares numerically
equal to one), using the formula above.

## Example

```python
# b=1, s=2048, h=4096, a=32
bytes_per_layer = activation_memory_bytes(1, 2048, 4096, 32)
# 2048*1*4096*(34 + 5*32*2048/4096) = 2048*4096*(34+80) = 956,301,312
```

## What the gate checks

The gate evaluates several `(b, s, h, a)` configurations spanning small and
large model shapes. The reference is computed independently, by summing the
two additive terms separately in bytes —

$$
\text{term}_1 = 34\,s\,b\,h, \qquad \text{term}_2 = 5\,a\,s^2\,b,
$$

$$
M_{\text{ref}} = \text{term}_1 + \text{term}_2 —
$$

rather than through the single fused expression above, so an implementation
that only gets the algebra right (not a copy of a particular arrangement)
will match. It compares your return value against this reference with

$$
\text{size\_ratio} = \left| \frac{M_{\text{candidate}}}{M_{\text{ref}}} - 1 \right|,
$$

which must satisfy $\text{size\_ratio} \le 10^{-9}$. Omitting the quadratic
attention-probability term, using the wrong byte width (e.g. 4 bytes instead
of the implied 16-bit activations), or swapping $s$ and $h$ in the formula
all produce a value far outside this tolerance.
