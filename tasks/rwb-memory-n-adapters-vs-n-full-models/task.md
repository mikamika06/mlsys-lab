## Context

In parameter-efficient fine-tuning (PEFT), a single **shared base model** of
dimension $d$ is frozen and a small **low-rank adapter** is trained per task.
A standard LoRA adapter on a layer contains a down-projection
$W_{\downarrow} \in \mathbb{R}^{d \times r}$ and an up-projection
$W_{\uparrow} \in \mathbb{R}^{r \times d}$, so one adapter contributes
$2dr$ parameters per layer.

Across $L$ layers, one adapter stores $2drL$ parameters.  For $N$ tasks we
keep one shared base and $N$ adapters, giving total parameters

$$P_{\text{adapter}} = P_{\text{base}} + N \cdot 2drL,$$

where $P_{\text{base}}$ is the total parameter count of the base model
($P_{\text{base}} = P_{\text{layer}} \cdot L$, with $P_{\text{layer}}$
given per layer).

The alternative is full fine-tuning: $N$ independent copies of the entire
base model, totaling

$$P_{\text{full}} = N \cdot P_{\text{base}} = N \cdot P_{\text{layer}} \cdot L.$$

Converting to bytes requires multiplying by the dtype size (e.g.\ $b = 2$ for
float16, $b = 4$ for float32).

The memory savings from adapters scale as $\frac{P_{\text{adapter}}}
{P_{\text{full}}} = \frac{1}{N} + \frac{2dr}{P_{\text{layer}}}$, which
shrinks both as $N$ grows and as the rank $r$ stays small relative to the
layer size.

## Task

Implement:

```python
def memory_comparison(
    d: int,
    r: int,
    num_layers: int,
    N: int,
    base_params_per_layer: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    """
    Return (adapter_strategy_bytes, full_copy_strategy_bytes).

    adapter_strategy_bytes:  1 shared base + N adapters (2*d*r params/layer each).
    full_copy_strategy_bytes: N full independent copies of the base.
    Both measured in bytes.
    """
```

All inputs are positive integers.  The returned values must be exact integers.

## Example

```python
>>> memory_comparison(
...     d=4096, r=64, num_layers=32, N=8,
...     base_params_per_layer=12_582_912, dtype_bytes=2
... )
(1073741824, 6442450944)
# 1 GiB for adapters vs 6 GiB for full copies
```

## What the gate checks

The gate verifies **exact integer match** across five test configurations,
including the example above, a tiny case, an edge case where $r = d$, a
degenerate case $N = 1$, and a large multi-adapter case.  A wrong formula for
the adapter parameter count, forgetting to include the shared base in one
strategy, or using float arithmetic that produces approximate results will all
fail.
