## Context

In many quantization schemes the total storage cost is a sum of two parts: the bits used to encode each individual value and the bits required for per‑block scaling parameters. For NF4 (four‑bit) quantization we store each weight as a 4‑bit integer together with an absolute maximum that scales the block. The absolute maximum is stored in floating‑point format.

If a tensor contains \(N\) parameters and we group them into blocks of size \(B\), then

$$
\text{total bits} = 4\,N + 32 \,\Bigl\lceil \frac{N}{B}\Bigr\rceil .
$$

The *bits per parameter* is therefore

$$
b_{\text{NF4}}(N,B)=\frac{4\,N+32\lceil N/B\rceil}{N}=4+\frac{32}{B}.
$$

When a second level of quantization (double‑quant) is added we first split the tensor into outer blocks of size \(B_1\). Each outer block stores its own 32‑bit absolute maximum. Inside each outer block we further split into inner blocks of size \(B_2\); each inner block stores an 8‑bit scaling factor. The resulting per‑parameter cost is

$$
b_{\text{NF4,\,double}}(N,B_1,B_2)=4+\frac{8}{B_1}+\frac{32}{B_1 B_2}.
$$

These formulas assume that \(N\) is large enough that the ceiling can be ignored; for the purposes of this task we use the exact integer division.

## Task

Implement `compute_nf4_bits(weights, block_size, outer_block, inner_block)`:

```python
def compute_nf4_bits(weights: list[float], block_size: int, outer_block: int, inner_block: int) -> tuple[float, float]:
    ...
```

The function receives a list `weights` (the tensor to be quantized). It must return a tuple `(bits_no_double, bits_with_double)` where

* `bits_no_double` is the NF4 bits per parameter without double‑quant,
* `bits_with_double` is the NF4 bits per parameter with double‑quant.

Both values should be computed exactly as described above and returned as Python floats (or Python scalars of type float64).

## Example

```python
weights = [random.gauss(0, 1) for _ in range(1000)] # 1 000 parameters
bits_no, bits_double = compute_nf4_bits(
    weights,
    block_size=64,
    outer_block=32,
    inner_block=8
)
print(bits_no, bits_double)
# → 4.5  4.75
```

The first number comes from \(4 + 32/64 = 4.5\).  
The second number comes from \(4 + 8/32 + 32/(32·8) = 4.75\).

## What the gate checks

* The returned tuple must contain two floats.
* Each value must match the reference calculation within a relative error of at most \(10^{-9}\).
* The implementation must not use any explicit Python loops; it should rely only on arithmetic operations and list properties.

The grader will generate several random weight tensors and compare your results against an oracle that recomputes the formulas. If any value deviates by more than the allowed tolerance, the gate fails.
