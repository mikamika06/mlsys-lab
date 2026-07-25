## Context

GELU and SiLU (Swish) are both smooth, sigmoid-shaped activations used
throughout modern transformer MLPs — often on the *same* tensor, in the
same forward pass. Writing them as two separate kernels means reading the
input tensor from global memory twice and paying two launch overheads for
work that's otherwise embarrassingly elementwise. **Fusing** them into one
kernel — read `x[i]` once, write both outputs from it — halves the input
traffic for free.

$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

$$
\mathrm{GELU}(x) \approx 0.5\, x \left(1 + \tanh\!\left(\sqrt{2/\pi}\,
\left(x + 0.044715\, x^3\right)\right)\right)
$$

Neither `erf` nor `tanh` exists as a builtin in this CUDA-C subset — build
`tanh` from `expf` instead, using the identity
$\tanh(z) = \dfrac{e^{2z} - 1}{e^{2z} + 1}$.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void fused_gelu_silu(float* gelu_out, float* silu_out, const float* x, int n);
```

For every `i` in `[0, n)`: load `x[i]` **once** into a local variable, then
write `silu_out[i] = x[i] / (1 + exp(-x[i]))` and
`gelu_out[i] = 0.5*x[i]*(1 + tanh(sqrt(2/pi)*(x[i] + 0.044715*x[i]^3)))`
(`tanh` built from `expf` as above; $\sqrt{2/\pi} \approx 0.7978845608$),
both from that one loaded value.

## Example

For `x = -1.0`: $\sigma(-1) \approx 0.26894$, so
$\mathrm{SiLU}(-1) \approx -1 \times 0.26894 \approx -0.26894$. The GELU
inner term is $0.7978845608 \times (-1 + 0.044715 \times (-1)^3) \approx
-0.83356$, $\tanh(-0.83356) \approx -0.68238$, giving
$\mathrm{GELU}(-1) \approx 0.5 \times (-1) \times (1 - 0.68238) \approx
-0.15881$.

## What the gate checks

`check.py` builds 64 random inputs in `[-4, 4]`, parses `solve.cu`, and
runs `fused_gelu_silu` on the software GPU (`arena.cuda_sim.GPU`) with a
2-block, 32-thread launch. It requires `max_abs_err <= 1e-5` against a
numpy reference (`np.tanh`-based, computed independently from the kernel's
`expf`-based `tanh`) across **both** output arrays — computing `SiLU`
correctly while leaving `gelu_out` untouched (or vice versa) still fails,
since the gate takes the max error over both outputs together.
