## Context

The **roofline model** connects an operation's arithmetic intensity to the hardware's peak capability. Arithmetic intensity is

$$AI = \frac{\text{FLOPs}}{\text{bytes transferred}}$$

and the machine's **ridge point** is

$$R = \frac{\text{peak FLOP/s}}{\text{peak BW (bytes/s)}}$$

An operation with $AI < R$ cannot saturate the compute units because data movement bottlenecks first: it is **memory-bound**. When $AI \ge R$ the operation can, in principle, keep the ALUs busy: it is **compute-bound**.

On a GPU, threads execute in warps of 32 lanes. A coalesced access pattern means consecutive threads touch consecutive addresses, so the memory system serves a warp in one or very few transactions. A correct classification kernel should exhibit coalesced reads and writes.

## Task

Write the CUDA-C kernel `classify(float* out, const float* ai, int n, float ridge)`.

**Global-memory layout** (total $2n$ floats):

| Range | Contents |
|---|---|
| `ai[0] … ai[n-1]` | Arithmetic intensity $AI_i$ of operation $i$ |
| `out[0] … out[n-1]` | Output: classification $c_i \in \{0.0,\;1.0\}$ |

For each thread `i = blockIdx.x * blockDim.x + threadIdx.x` with `i < n`:

1. Read `ai[i]`.
2. Classify: `out[i] = 0.0f` if `ai[i] < ridge` (memory-bound), else `out[i] = 1.0f` (compute-bound).

Keep the access coalesced: thread `i` must touch address `i` in both `ai` and `out` (no stride, no permutation).

## Example

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = (ai[i] >= ridge) ? 1.0f : 0.0f;
}
```

A small case with $n = 4$, $\text{ridge} = 4.0$: `ai = [2.0, 6.0, 3.5, 8.0]` produces `out = [0.0, 1.0, 0.0, 1.0]`.

## What the gate checks

The grader parses your kernel with the real CUDA-C interpreter and launches it on the software GPU, then compares the `out` slice against a NumPy reference computed with `np.where(ai >= ridge, 1.0, 0.0)`.

| Metric | Condition | Meaning |
|---|---|---|
| `max_abs_err` | $\le 10^{-9}$ | Every classification matches the reference |
| `transactions` | $\le 50$ | Global-memory access is coalesced (a strided or permuted access pattern blows past this) |
