## Context

LLM inference has two very different phases. **Prefill** processes the
whole prompt (length $P$) in one batched forward pass: every linear
layer's weights are loaded from HBM once and reused across all $P$
tokens. **Decode** generates one token at a time: the same weights are
loaded again for a single token, plus the growing KV cache must be read.

The roofline model classifies a computation by its arithmetic intensity
$\mathrm{AI} = \text{FLOPs}/\text{bytes}$ against the hardware's ridge
point $\rho = \text{peak\_FLOPs/s} \,/\, \text{peak\_bytes/s}$: a phase
with $\mathrm{AI} \ge \rho$ is **compute-bound**, otherwise it is
**bandwidth-bound**. Because prefill's compute grows quadratically with
$P$ while its weight traffic is paid only once, prefill tends toward
compute-bound as $P$ grows. Decode pays the full weight-loading cost for
just one token's worth of work, so it tends to be bandwidth-bound.

For one transformer layer with hidden size $H$ (attention heads $N_h$
cancel out of every formula below — they only redistribute the same
$H^2$ work across heads), let $D$ = bytes per scalar (e.g. 2 for fp16, 4
for fp32). The linear layers (QKV, output projection, and an FFN
$H\!\to\!4H\!\to\!H$) together hold $12H^2$ weight scalars, so

$$
\text{weight\_bytes} = 12H^2 D .
$$

**Prefill** ($P$ tokens, one pass; attention modeled as the full — not
causally masked — $P\times P$ score matrix):

$$
\text{FLOPs}_{\text{pre}} = 12H^2P + 2HP^2, \qquad
\text{bytes}_{\text{pre}} = \text{weight\_bytes} + 3DPH ,
$$

the $3DPH$ covering reading the $P$ input embeddings ($DPH$) and writing
the $P$ new rows into the KV cache ($2DPH$).

**Decode** (1 new token, attending to $S=P$ cached tokens from the
prefill that just finished):

$$
\text{FLOPs}_{\text{dec}} = 12H^2 + 2HS, \qquad
\text{bytes}_{\text{dec}} = \text{weight\_bytes} + D(2SH + 3H) ,
$$

the $2DSH$ covering reading the cached $K,V$ and the trailing $3DH$
covering the new input token plus writing its own new $K,V$ row.

$$
\mathrm{AI} = \text{FLOPs}/\text{bytes}, \qquad
\rho = \text{peak\_flops\_per\_s} \,/\, \text{peak\_bytes\_per\_s} .
$$

## Task

Implement `prefill_decode_roofline`:

```python
def prefill_decode_roofline(hidden_size: int,
                             num_heads: int,
                             prefill_len: int,
                             dtype_bytes: int,
                             peak_flops_per_s: float,
                             peak_bytes_per_s: float) -> dict:
    ...
```

Return a `dict` with keys `"prefill"` and `"decode"`, each mapping to a
`dict` with keys `"flops"`, `"bytes"`, `"ai"` (all floats, using the
closed-form expressions above) and `"roofline_class"` — the string
`"compute-bound"` if $\mathrm{AI} \ge \rho$, else `"bandwidth-bound"`.

## Example

```python
result = prefill_decode_roofline(
    hidden_size=4096, num_heads=32, prefill_len=2048,
    dtype_bytes=2, peak_flops_per_s=20e12, peak_bytes_per_s=800e9,
)
print(result["prefill"]["roofline_class"])  # "compute-bound"
print(result["decode"]["roofline_class"])   # "bandwidth-bound"
```

## What the gate checks

The grader recomputes `flops`, `bytes` and `ai` for both phases from the
same closed-form expressions, for several random `(hidden_size,
prefill_len, dtype_bytes)` configurations and hardware profiles
(`peak_flops_per_s`, `peak_bytes_per_s`).

* **`modeled_arith_intensity`** — the maximum relative error between your
  6 returned numeric fields (`flops`/`bytes`/`ai` for each phase) and the
  reference. Must satisfy $\le 10^{-6}$ (this is closed-form arithmetic,
  so a correct implementation matches to float64 precision).
* **`classification_exact`** — `1.0` only if every `roofline_class` you
  return matches the reference classification (computed by comparing the
  *reference* AI to the ridge point $\rho$ for that trial's hardware
  profile), else `0.0`.
