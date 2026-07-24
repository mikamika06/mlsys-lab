## Context

A real FP8 KV cache almost never uses one global scale for the whole
tensor — different attention heads see very different activation ranges,
so a single scale either clips the loud heads or wastes precision on the
quiet ones. The standard fix is **per-head quantization**: derive one
amax-based scale *per head*, quantize that head's K and V independently
with it, and only then run attention.

For a head's raw values $x$, the E4M3FN (1 sign, 4 exponent, 3 mantissa
bit, bias 7) quantizer works as:

$$
\text{scale}_h = \frac{\max_i |x_{i,h}|}{\text{FP8\_MAX}}, \qquad
\text{code}_{i,h} = \mathrm{encode}_{\mathrm{e4m3}}\!\left(\frac{x_{i,h}}{\text{scale}_h}\right),
$$

where $\text{FP8\_MAX} = 448$ is the largest finite magnitude representable
in E4M3FN, and $\mathrm{encode}_{\mathrm{e4m3}}$ rounds to the nearest
representable code (ties to even). Dequantization reverses it:
$\tilde x_{i,h} = \mathrm{decode}_{\mathrm{e4m3}}(\text{code}_{i,h}) \cdot \text{scale}_h$.

## Task

Implement `per_head_kv_attention`:

```python
def per_head_kv_attention(K, V, q):
    ...
```

- `K`, `V` — `float64` arrays of shape $(S, H, D)$: raw (unquantized) keys
  and values, $S$ sequence positions, $H$ heads, head dim $D$.
- `q` — `float64` array of shape $(H, D)$: one query vector per head.

You must, **per head** $h$:

1. Compute $\text{k\_scale}_h = \max_i |K_{i,h,:}| / 448$ and
   $\text{v\_scale}_h = \max_i |V_{i,h,:}| / 448$ (independent scales for
   K and V).
2. Quantize $K_{:,h,:}$ and $V_{:,h,:}$ to E4M3FN codes using that head's
   scale, then dequantize back to float.
3. Run standard scaled dot-product attention over the dequantized
   $\tilde K_h, \tilde V_h$ for that head:
   $$
   o_h = \sum_i \mathrm{softmax}\!\left(\frac{q_h^\top \tilde K_h}{\sqrt D}\right)_{\!i} \tilde V_{h,i} .
   $$

Return a `float64` array of shape $(H, D)$ — one output vector per head.

## Example

If head 0 has values in $[-2, 2]$ and head 1 has values in $[-200, 200]$,
a single shared scale would either clip head 1 badly or leave head 0 with
almost no usable mantissa bits. With per-head scales, $\text{k\_scale}_0
\approx 2/448$ and $\text{k\_scale}_1 \approx 200/448$, and each head
quantizes against its own range.

## What the gate checks

The grader builds several $(S, H, D)$ cases where different heads are
deliberately scaled to very different magnitudes, computes the real
per-head amax scale, E4M3FN-encodes and decodes both K and V per head
(round-to-nearest-even, real bit layout), and runs attention — all
independently of your code. Your output is compared against that oracle:

$$
\max_i |o_i - \hat o_i| \le 10^{-6}.
$$

Using one global scale instead of per-head scales, swapping the K/V
scales, using the wrong FP8 max (448), or mis-decoding the E4M3FN bit
layout will all produce a large, easily detected deviation.
