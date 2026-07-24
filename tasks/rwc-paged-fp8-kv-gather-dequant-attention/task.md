## Context

A production KV cache composes two techniques you've likely met
separately: **paging** (physical storage split into fixed-size blocks,
addressed indirectly through a per-sequence block table) and **FP8
quantization** (each cached key/value stored as a 1-byte E4M3FN code, with
one dequantization scale per head). Reconstructing attention output from
such a cache means correctly composing *both*:

1. **Gather**: physical blocks $K_{\mathrm{phys}}, V_{\mathrm{phys}} \in
   \{0,\dots,255\}^{B \times S \times H \times D}$ (raw E4M3FN byte codes,
   $H$ heads, head dim $D$) are reordered into logical order via a block
   table $T \in \mathbb{N}^{L_b}$, then **truncated to the true sequence
   length** $n$ — the last logical block may be only partially filled;
   anything at or past $n$ is stale.
2. **Dequantize**: each logical code is decoded per the real E4M3FN
   bit-pattern formula (1 sign, 4 exponent, 3 mantissa bits, bias 7,
   subnormals via the exponent-field-0 branch) and multiplied by that
   **head's** scale:
   $$
   \tilde k_{i,h} = \mathrm{decode}_{\mathrm{e4m3}}(\text{code}_{i,h}) \cdot \text{k\_scale}_h .
   $$
3. **Attend**: standard per-head scaled dot-product attention over the
   dequantized, truncated logical sequence,
   $$
   o_h = \sum_i \mathrm{softmax}\!\left(\frac{q_h^\top \tilde K_h}{\sqrt D}\right)_{\!i} \tilde V_{h,i} .
   $$

## Task

Implement `paged_fp8_attention`:

```python
def paged_fp8_attention(k_codes_phys, v_codes_phys, k_scale, v_scale, block_table, seq_len, q):
    ...
```

- `k_codes_phys`, `v_codes_phys` — `uint8` arrays of shape $(B, S, H, D)$:
  raw E4M3FN byte codes in physical block layout. Physical blocks not
  referenced by `block_table` may hold unrelated leftover codes.
- `k_scale`, `v_scale` — `float` arrays of shape $(H,)$: per-head
  dequantization scales.
- `block_table` — array of shape $(L_b,)$: physical block index per
  logical position, in logical order.
- `seq_len` — int, the true number of valid logical tokens
  $n \le L_b \cdot S$; only the first `seq_len` gathered rows are valid.
- `q` — `float` array of shape $(H, D)$: one query vector per head.

Return a `float64` array of shape $(H, D)$.

## Example

With one logical block ($L_b=1$, $S=4$) and `seq_len=3`, the 4th row of
whatever physical block backs that logical block is stale and must be
excluded from both the gather and the attention softmax, exactly as with
an unquantized paged cache — only now every value that *does* count first
passes through the E4M3FN decode formula and its head's scale.

## What the gate checks

The grader builds a real K/V sequence, quantizes it to E4M3FN codes with a
genuine per-head max-abs scale, scatters it into a shuffled pool of
physical blocks (extra unused physical slots hold distinct garbage codes,
and the true sequence length leaves the last logical block partially
filled), and compares your output against the same gather → truncate →
dequantize → attend pipeline computed independently:

$$
\max_i |o_i - \hat o_i| \le 10^{-6}.
$$

Skipping the `block_table` indirection, forgetting to truncate to
`seq_len`, dequantizing with the wrong head's scale, or misreading the
E4M3FN bit layout will all produce a large, easily detected deviation.
