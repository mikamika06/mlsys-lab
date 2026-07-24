## Context

Dropout randomly zeroes out elements of an activation vector during training to prevent overfitting. **Inverted** dropout rescales the elements that survive by $\frac{1}{1-p}$, so the *expected* value of each element is unchanged whether or not dropout is active. Generating the random keep/drop decisions is a separate concern from applying the scaling correctly -- here the keep-mask is already computed for you (as a 0.0/1.0 array), and the kernel's job is purely the scaling arithmetic.

## Task

Write the CUDA-C kernel `dropout(float* out, const float* x, const float* mask, int n, float p)`.

**Global-memory layout** (three arrays of length $n$ each):

| Array | Contents |
|---|---|
| `x[0..n-1]` | input activations |
| `mask[0..n-1]` | precomputed keep-mask: `1.0` = keep, `0.0` = drop |
| `out[0..n-1]` | output (yours to fill in) |

For each thread `i = blockIdx.x * blockDim.x + threadIdx.x` with `i < n`:

$$\text{out}[i] = \text{mask}[i] \cdot \frac{x[i]}{1 - p}$$

Keep the access coalesced: thread `i` must touch address `i` in `x`, `mask`, and `out`.

## Example

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = mask[i] * x[i] / (1.0f - p);
}
```

With `x = [0.5, -1.2, 3.0]`, `mask = [1.0, 0.0, 1.0]`, `p = 0.25`: `out = [0.6667, 0.0, 4.0]`.

## What the gate checks

The grader parses your kernel with the real CUDA-C interpreter and launches it on the software GPU, then compares `out` against `mask * x / (1 - p)` computed with NumPy.

| Metric | Condition | Meaning |
|---|---|---|
| `max_abs_err` | $\le 10^{-9}$ | Every output element matches the reference exactly |
| `transactions` | $\le 80$ | Global-memory access is coalesced |

Zeroing dropped elements but forgetting to rescale the kept ones (a very common real bug -- "dropout" without the "inverted" part) throws off every kept element's value.
