## Context

Storing a dropout mask from the forward pass just to reuse it in the
backward pass costs one full extra buffer the size of the activation —
real training frameworks skip it. Instead, both passes derive the *same*
keep/drop decision from a pure, stateless function of `(seed, element
index)`: the forward pass computes it once to apply dropout, and the
backward pass computes it *again*, from scratch, with no buffer passed
between them. As long as both passes hash the exact same `(seed, index)`
pair the exact same way, the masks are guaranteed identical — determinism
by construction, not by storage.

The trap is the index itself: on a multi-block launch, `threadIdx.x`
alone repeats across blocks (`0..31` in every block), so hashing with
`threadIdx.x` instead of the true global index
`blockIdx.x*blockDim.x + threadIdx.x` would alias every block's element 0
onto the same mask value — correct-looking within one block, silently
wrong the moment there's more than one.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void dropout_fwd_bwd(float* fwd_out, float* bwd_grad, const float* x,
                                 const float* grad_in, float seed, float keep_prob, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`: compute
`h = i`, run 3 rounds of `h = (h*48271 + seed + r*7919) mod 1000003`
(`r = 0,1,2`), then `rand01 = h / 1000003`. `keep = 1` if `rand01 <
keep_prob` else `0`. Set `fwd_out[i] = keep * x[i] / keep_prob`. Then
**recompute** `h`/`rand01`/`keep` the exact same way a second time (not
reusing the first result) to set `bwd_grad[i] = keep * grad_in[i] /
keep_prob`.

## Example

At `i=5`, `seed=777`: the hash produces some `rand01` value; whichever
side of `keep_prob=0.7` it falls on decides both `fwd_out[5]` (scaling
`x[5]`) and `bwd_grad[5]` (scaling `grad_in[5]`) — recomputed
independently, but from identical inputs, so they must land on the same
side every time.

## What the gate checks

`max_abs_err <= 1e-9` on both `fwd_out` and `bwd_grad` over a 64-element,
2-block launch (`seed=777`, `keep_prob=0.7`), against a numpy oracle
running the same hash. Hashing with `threadIdx.x` instead of the global
index `i` produces correct results in block 0 but silently wrong ones in
block 1 (aliased onto block 0's mask) — caught by this task using more
than one block on purpose. Any mismatch between the forward hash and the
backward re-hash (different round count, different seed handling) also
fails the match.
