## Context

Pipeline parallelism splits a deep neural network across multiple accelerators.  
A minibatch is further divided into **microbatches** that flow through the pipeline stages.

**GPipe** (Huang et al., 2019) sends all microbatches forward before any backward pass begins.  
At steady state, the peak activation memory is

\[
\text{GPipe}(M, m) = M \cdot m \cdot a \cdot d,
\]

where $M$ is the number of microbatches, $m$ the per-microbatch activation size,  
$a$ the number of activations stored per microbatch per pipeline stage, and $d$ the number of pipeline stages.

**1F1B** (one-forward-one-backward, Narayanan et al., 2019) interleaves forward and backward passes.  
Its peak activation memory is

\[
\text{1F1B}(p, m) = (p - 1) \cdot m \cdot a \cdot d,
\]

where $p$ is the pipeline depth (number of stages).

## Task

Implement `compute_peak_activation_bytes(schedule, M, p, a_bytes, d)`:

```python
def compute_peak_activation_bytes(schedule: str, M: int, p: int, a_bytes: int, d: int) -> int:
    ...
```

Arguments:
- `schedule`: either `"gpipe"` or `"1f1b"`.
- `M`: number of microbatches (for GPipe).
- `p`: pipeline depth / number of stages (for 1F1B).
- `a_bytes`: bytes of one activation tensor per microbatch per stage.
- `d`: number of pipeline stages (for both — in 1F1B it's the same as `p` and must equal `p`; provided for uniform interface).

For GPipe return $M \cdot a\_bytes \cdot d$.  
For 1F1B return $(p - 1) \cdot a\_bytes \cdot d$.

## Example

```python
>>> compute_peak_activation_bytes("gpipe", M=8, p=4, a_bytes=1024, d=4)
32768   # 8*1024*4
>>> compute_peak_activation_bytes("1f1b", M=8, p=4, a_bytes=1024, d=4)
12288   # (4-1)*1024*4
```

## What the gate checks

Your `compute_peak_activation_bytes` is called on 6 random parameter sets
for each schedule (12 total). The gate `size_ratio_pipe` checks that the
returned integer exactly equals the analytic model computed by the grader.
A single wrong value yields `size_ratio_pipe = 0.0`.
