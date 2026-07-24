## Context

The **roofline model** relates a kernel's floating-point throughput to its data
movement. The key quantity is **arithmetic intensity** (AI):

$$\mathrm{AI} = \frac{\text{FLOPs}}{\text{bytes moved}}$$

A kernel is *memory-bound* when $\mathrm{AI}$ is below the machine's
balance point $\beta = \frac{\text{peak FLOP/s}}{\text{peak byte/s}}$,
and *compute-bound* otherwise.

Two canonical kernels illustrate this contrast.

**SAXPY** ($Y \leftarrow aX + Y$, vectors of length $n$, FP64):

- Each element: one multiply + one add $\Rightarrow 2n$ FLOPs.
- Data movement: read $X$ ($8n$ bytes), read $Y$ ($8n$ bytes), write
  $Y$ ($8n$ bytes) $\Rightarrow 24n$ bytes. The scalar $a$ is loaded once
  and is negligible; the standard convention ignores it.

$$\mathrm{AI}_{\text{SAXPY}} = \frac{2n}{24n} = \frac{1}{12} \approx 0.083$$

This is constant regardless of $n$ — SAXPY is always memory-bound.

**GEMM** ($C \leftarrow AB$, with $A \in \mathbb{R}^{m \times k}$,
$B \in \mathbb{R}^{k \times n}$, all FP64):

- Each of the $mn$ output elements accumulates $k$ multiply-adds
  $\Rightarrow 2mnk$ FLOPs.
- Data movement: read $A$ ($8mk$ bytes), read $B$ ($8kn$ bytes),
  write $C$ ($8mn$ bytes) $\Rightarrow 8(mk + kn + mn)$ bytes.

$$\mathrm{AI}_{\text{GEMM}} = \frac{2mnk}{8(mk + kn + mn)}$$

For square $n \times n$ matrices this simplifies to $\frac{n}{12}$, which
grows with $n$ — large GEMMs are compute-bound.

## Task

Implement `compute_roofline_metrics`:

```python
def compute_roofline_metrics(op: str, **dims) -> dict:
    ...
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `op` | `str` | `'saxpy'` or `'gemm'` |
| `n` | `int` | (SAXPY) vector length |
| `m`, `k`, `n` | `int` | (GEMM) matrix dimensions: $A$ is $m \times k$, $B$ is $k \times n$ |

**Returns**

A dict with three `float` values:

- `'flops'` — total floating-point operations.
- `'bytes'` — total data-movement bytes (FP64, 8 bytes per element;
  ignore scalar operands).
- `'ai'` — arithmetic intensity $= \text{flops} / \text{bytes}$.

All inputs are positive integers. Use FP64 (8 bytes per element) for the
byte count.

## Example

```python
compute_roofline_metrics('saxpy', n=12)
# {'flops': 24.0, 'bytes': 288.0, 'ai': 0.083333...}

compute_roofline_metrics('gemm', m=2, k=3, n=4)
# {'flops': 48.0, 'bytes': 184.0, 'ai': 0.260869...}
```

## What the gate checks

Three gates, all using relative error
$\mathrm{rel\_err} = \frac{\lVert \hat{x} - x \rVert}{\lVert x \rVert + \epsilon}$
against an oracle that recomputes the formulas independently:

1. **flops_rel_err** $< 10^{-6}$ — correct FLOP count.
2. **bytes_rel_err** $< 10^{-6}$ — correct byte count.
3. **ai_rel_err** $< 10^{-6}$ — correct arithmetic intensity.

Five cases are tested: two SAXPY sizes and three GEMM shapes (including
a degenerate $1 \times 1 \times 1$ case). All three gates must pass.
