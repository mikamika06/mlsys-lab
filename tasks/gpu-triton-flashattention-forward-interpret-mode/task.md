## Context

Ordinary softmax attention computes a full row of scores
$s_j = Q_i \cdot K_j$ for every key $j$, finds its max (for numerical
stability), exponentiates, normalizes, and only then multiplies by
$V$ — which means materializing an entire length-$N$ row before any of
it can be used. FlashAttention's trick is an **online softmax**: update
a running max, running normalizer, and running weighted output
*incrementally*, one key at a time, so the full row of scores never has
to exist anywhere at once.

The recurrence handles the case where a *later* key turns out to have a
higher score than everything seen so far — which would ordinarily mean
going back and rescaling every earlier term. Instead of doing that
after the fact, it corrects incrementally: whenever the running max
$m$ increases to $m'$, every accumulated quantity so far gets multiplied
by $\exp(m - m')$ — shrinking the old, now-relatively-smaller terms by
exactly the right factor — before the new key's contribution is added
in. Run this to completion over all $N$ keys and the result is
*exactly* the same number ordinary softmax attention would produce,
never having stored more than one key/value pair's worth of extra state
at a time.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void flash_attention_fwd(const float* Q, const float* K, const float* V,
                                     float* O, int N, float scale);
```

One thread per query row `i` (head dim fixed at 4). Initialize
`m = -1e30`, `l = 0`, `acc0..acc3 = 0`. For each key `j` from `0` to
`N-1`:

1. `score = (Q[i][0]*K[j][0] + ... + Q[i][3]*K[j][3]) * scale`.
2. `new_m = max(m, score)`, `correction = exp(m - new_m)`,
   `p = exp(score - new_m)`.
3. `l = l*correction + p`.
4. For each `d`: `acc_d = acc_d*correction + p*V[j][d]`.
5. `m = new_m`.

After the loop, write `O[i][d] = acc_d / l` for each `d`.

## Example

Two keys, scores `[1.0, 3.0]` (second key more relevant). After key 0:
`m=1.0, l=1.0, acc=V[0]`. Key 1: `new_m=3.0`,
`correction=exp(1-3)=0.135`, `p=exp(3-3)=1.0`; `l = 1.0*0.135 + 1.0 =
1.135`; `acc = V[0]*0.135 + 1.0*V[1]`. Dividing by `l` gives the same
answer as computing `softmax([1,3])` directly and weighting `V` by
it — `0.135/1.135 ≈ 0.119` and `1.0/1.135 ≈ 0.881`, matching
`softmax([1,3]) ≈ [0.119, 0.881]` to the last digit.

## What the gate checks

The grader launches `flash_attention_fwd` on fixed `16x4` `Q`, `K`, `V`
and compares the output against ordinary (full-row) softmax attention
computed directly in numpy. It requires

$$
\mathrm{rel\_err} \le 10^{-6}
$$

Online softmax isn't an approximation — done correctly, it reproduces
ordinary softmax attention's output to within ordinary floating-point
rounding, here landing around `5\times10^{-16}$, at the machine-precision
floor.
