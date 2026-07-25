## Context

Real dropout does two things per element, not one: decide keep-or-drop
(the "mask"), and rescale whatever survives. Just zeroing dropped elements
and leaving the rest alone shrinks the *expected* sum of the activations
by a factor of `(1 - p)` — every downstream layer would need to know
dropout ran, and by how much, just to compensate. **Inverted scaling**
divides every surviving element by `(1 - p)` right there in the dropout
kernel, so $E[\text{out}[i]] = x[i]$ regardless of `p` — nothing
downstream has to change between training (dropout on) and inference
(dropout off, an identity op).

Like any GPU dropout, the keep/drop decision has to be a pure function of
`(seed, index)` — reproducible from any launch geometry, no state shared
between threads. This subset has no RNG builtin and no bitwise operators,
so the hash is pure multiplication/modulo:

$$
h = (\text{seed} + i \times 2654435761) \bmod 1000000007 \qquad
u = h / 1000000007
$$

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void triton_dropout(float* out, const float* x, int n, int seed, float p);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x` in `[0, n)`: compute `h` and
`u` as above. If `u < p`, `out[i] = 0.0f` (dropped). Otherwise,
`out[i] = x[i] / (1.0f - p)` (kept, rescaled).

## Example

With `seed = 999`, `p = 0.4`: an element whose hash gives `u < 0.4` comes
out `0.0`, full stop, regardless of its input value. An element with
`u >= 0.4` comes out `x[i] / 0.6` — about `1.667x` its original value, not
`x[i]` unchanged, because the elements that *would* have balanced it out
on average just got zeroed.

## What the gate checks

`check.py` recomputes the same hash and rescaling independently in Python
for `n = 200` random inputs, parses `solve.cu`, and runs `triton_dropout`
on the software GPU (`arena.cuda_sim.GPU`) with a 7-block, 32-thread launch
(224 threads > 200, exercising the tail guard too). Both the seed and the
hash are fixed, so the whole computation is deterministic — an exact,
per-element `max_abs_err <= 1e-6` is the right test here, not a looser
distributional one. Applying the mask but skipping the `/ (1 - p)` rescale
matches on every dropped element (`0.0` either way) but is off by a
consistent `1.667x` factor on every surviving one.
