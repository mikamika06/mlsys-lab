## Context

Mixed-precision training with Adam keeps several buffers resident on the
GPU per parameter: the low-precision parameter and gradient used for compute
(e.g. `float16`/`bfloat16`), plus the optimizer's own state — an `float32`
master copy of the weights and Adam's two running-average tensors, $m$ and
$v$. On top of all of that, activation memory from the forward/backward pass
takes a further fixed chunk of GPU memory that does **not** scale with
whether you offload the optimizer or not.

ZeRO-Offload-style training moves the optimizer state — the master weights,
$m$, and $v$ — to CPU memory and only ever moves the (small) low-precision
gradient over to run the optimizer step there, leaving only the
low-precision parameters, gradients, and the activations on the GPU.

For a model with $n$ parameters, let $p, g, m_b, v_b, w_b$ be the
per-parameter byte sizes of the parameter, gradient, master weight, Adam
$m$, and Adam $v$ tensors respectively, and let $C$ be the fixed activation
memory in bytes. The GPU memory footprint **before** offloading is

$$
T_{\text{before}} = n \cdot (p + g + w_b + m_b + v_b) + C
$$

and the number of bytes moved off the GPU by offloading the optimizer state is

$$
O = n \cdot (w_b + m_b + v_b)
$$

The fractional VRAM reduction is simply the offloaded bytes over the
original total:

$$
\text{reduction} = \frac{O}{T_{\text{before}}}
$$

## Task

Implement `vram_reduction_from_offload`:

```python
def vram_reduction_from_offload(
    n_params: int,
    param_bytes: int,
    grad_bytes: int,
    master_bytes: int,
    m_bytes: int,
    v_bytes: int,
    activation_bytes: int,
) -> float:
    ...
```

* `n_params` — number of model parameters.
* `param_bytes`, `grad_bytes` — bytes per parameter for the compute-precision
  parameter and gradient tensors that stay on the GPU.
* `master_bytes`, `m_bytes`, `v_bytes` — bytes per parameter for the
  optimizer-state tensors (`float32` master weights, Adam `m`, Adam `v`)
  that get offloaded to CPU.
* `activation_bytes` — a fixed number of GPU bytes for activations, unaffected
  by offloading.

Return the fractional VRAM reduction `reduction = O / T_before` as defined
above (a `float` in $(0, 1)$).

## Example

```python
n_params = 1_000_000_000        # 1B params
param_bytes = 2                 # fp16 param
grad_bytes = 2                  # fp16 grad
master_bytes = 4                # fp32 master weight
m_bytes = 4                     # fp32 Adam m
v_bytes = 4                     # fp32 Adam v
activation_bytes = 2_000_000_000  # 2 GB of activations

vram_reduction_from_offload(n_params, param_bytes, grad_bytes,
                             master_bytes, m_bytes, v_bytes, activation_bytes)
# T_before = 1e9 * (2+2+4+4+4) + 2e9 = 16e9 + 2e9 = 18e9
# O        = 1e9 * (4+4+4)          = 12e9
# reduction = 12e9 / 18e9 ≈ 0.6667
```

## What the gate checks

A single **rel_err** gate tries several random combinations of `n_params`,
byte sizes, and `activation_bytes`, computes the reference `reduction` with
the closed-form formula above, and requires your returned value to match
within a relative error of `1e-9`. Any exception or non-numeric return fails
the gate.
