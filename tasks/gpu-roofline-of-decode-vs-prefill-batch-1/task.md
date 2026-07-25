## Context

Autoregressive LLM inference runs the exact same weight matrices two
completely different ways:

- **Decode** (batch = 1): each new token is a single matrix-*vector*
  product against a `d_in x d_out` weight. The weight matrix has to be
  loaded from memory in full — and it dominates the byte traffic — but
  it's only ever used to produce ONE output vector. FLOPs $=
  2 \cdot d_{in} \cdot d_{out}$, bytes $\approx 4(d_{in} d_{out} + d_{in}
  + d_{out})$. As the matrix gets bigger, both numerator and denominator
  scale the same way ($d_{in} d_{out}$ dominates both), so arithmetic
  intensity converges to a **fixed constant around 0.5 FLOPs/byte** —
  decode is memory-bound no matter how large the model is.
- **Prefill** (batch = $t$ tokens, processed together): the SAME weight
  matrix, loaded ONCE, reused across all $t$ tokens in a real
  matrix-*matrix* product. FLOPs $= 2t \cdot d_{in} d_{out}$, bytes
  $\approx 4(d_{in} d_{out} + t \cdot d_{in} + t \cdot d_{out})$ — the
  weight's byte cost gets amortized across $t$ tokens' worth of FLOPs, so
  arithmetic intensity grows with $t$.

Same weights, same hardware — the only thing that changed is how many
tokens share one weight load. That's why decode-heavy autoregressive
serving is fundamentally memory-bandwidth-limited while prefill (or big
batches) can approach the hardware's peak FLOP rate.

## Task

Write a CUDA-C kernel (single thread):

```cpp
__global__ void decode_prefill_ai(float* out, float d_in, float d_out, float t,
                                    float peak_flops, float peak_bw);
```

Compute `ridge = peak_flops / peak_bw`, then:

- `out[0] = decode_ai` = $2 d_{in} d_{out} \;/\; 4(d_{in} d_{out} +
  d_{in} + d_{out})$
- `out[1] = prefill_ai` = $2 t \, d_{in} d_{out} \;/\; 4(d_{in} d_{out} +
  t\,d_{in} + t\,d_{out})$
- `out[2] = 1.0` if `decode_ai >= ridge` else `0.0`
- `out[3] = 1.0` if `prefill_ai >= ridge` else `0.0`

## Example

With `d_in = d_out = 4096`, `t = 128`, `peak_flops = 1000`,
`peak_bw = 100` (ridge $= 10.0$):

```
decode_ai  ≈ 0.499756   -> memory-bound (0.0)
prefill_ai ≈ 60.235294  -> compute-bound (1.0)
```

Decode sits at essentially the theoretical $0.5$ floor regardless of how
large `d_in`/`d_out` get — batching 128 tokens together into one prefill
pass amortizes the exact same weight load across $120\times$ more
arithmetic intensity, crossing well past the ridge point into
compute-bound territory.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU with the fixed sizes above, requiring
`max_abs_err <= 1e-6` against all 4 values computed directly in Python.
Using `2*d_in*d_out` for prefill's FLOPs too (forgetting the `t`
multiplier — a plausible copy-paste from the decode formula) undercounts
prefill's arithmetic intensity by roughly $60\times$ and likely
misclassifies it as memory-bound, failing the gate on 3 of the 4 values
at once.
