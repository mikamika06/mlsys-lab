## Context

ZeRO (Zero Redundancy Optimizer) shards the memory footprint of a model across multiple GPUs. For a model with $\Phi$ parameters, each parameter is stored as an fp16 value ($2$ bytes). The gradient for every parameter is also fp16. Adam uses three fp32 buffers per parameter: a master copy of the parameter and two momentum terms $m$ and $v$, totalling $12$ bytes.

The memory required on one GPU depends on the ZeRO stage:

- **Stage 0** – no sharding; each GPU holds all parameters, gradients and optimizer state.
- **Stage 1** – only the optimizer buffers are partitioned across $N$ GPUs.
- **Stage 2** – both parameters/gradients and optimizer buffers are partitioned. In practice ZeRO‑2 keeps a full copy of the optimizer on every rank.

Thus for a given $\Phi$, number of GPUs $N$ and stage $s\in\{0,1,2\}$ the per‑GPU byte count is

$$
B(\Phi,N,s)=
\begin{cases}
16\,\Phi & s=0\\[4pt]
4\,\Phi + \dfrac{12\,\Phi}{N} & s=1\\[6pt]
\dfrac{4\,\Phi}{N}+12\,\Phi & s=2~.
\end{cases}
$$

The task is to implement this formula.

## Task

Implement the function `per_gpu_bytes(phi: int, n_gpus: int, stage: int) -> int` that returns the exact number of bytes a single GPU must allocate for parameters, gradients and Adam optimizer state according to the ZeRO memory accounting described above. The function should raise a `ValueError` if `stage` is not 0, 1 or 2.

## Example

```python
>>> per_gpu_bytes(10_000_000, 8, 2)
12500000
```

Explanation: $\Phi=10\,\text{M}$, $N=8$, stage 2 gives  
$B = \frac{4\Phi}{N} + 12\Phi = \frac{40\,\text{M}}{8}+120\,\text{M}=5\,\text{M}+120\,\text{M}=125\,\text{M}$ bytes.

## What the gate checks

The grader computes a reference value using the same formula and compares it to your output with exact integer equality (`==`). No other metrics are evaluated. A correct implementation passes; any deviation causes the gate to fail.
