## Context

During transformer training, activation checkpointing reduces memory by storing only
selected boundary hidden states and recomputing intermediate values during the
backward pass. A simplified checkpoint memory model stores one hidden state per
checkpoint segment.

For a sequence length $L$, batch size $B$, hidden size $H$, and element size
$s$ bytes, the activation bytes for a standard checkpoint layout can be modeled as

$$
M_{\mathrm{standard}} = B \cdot L \cdot H \cdot s .
$$

An offloaded checkpoint implementation keeps the same logical boundary states but
moves those stored states to CPU memory. The GPU retains only a small fraction of
the checkpoint state while the CPU holds the full offloaded copy. In this task,
the GPU activation footprint for offloaded checkpointing is

$$
M_{\mathrm{offloaded}} = B \cdot H \cdot s .
$$

Given a fixed GPU activation budget $G$, the maximum supported context length is

$$
L_{\mathrm{max}} = \frac{G}{B \cdot H \cdot s}.
$$

The context multiplier measures how much longer a sequence can be trained when
using offloaded checkpoint states instead of standard checkpointing:

$$
\mathrm{multiplier}
=
\frac{L_{\mathrm{offloaded}}}{L_{\mathrm{standard}}}
=
\frac{M_{\mathrm{standard}}}{M_{\mathrm{offloaded}}}.
$$

## Task

Implement `context_multiplier`:

```python
def context_multiplier(
    batch_size: int,
    hidden_size: int,
    element_bytes: int,
    gpu_budget_bytes: int,
    sequence_length: int,
) -> float:
    ...
```

Return the ratio between the maximum context length possible with offloaded
checkpointing and the maximum context length possible with standard checkpointing
under the same GPU activation budget.

The function should use the activation memory model above. Return a Python
`float`.

## Example

```python
x = context_multiplier(
    batch_size=2,
    hidden_size=4096,
    element_bytes=2,
    gpu_budget_bytes=2_000_000_000,
    sequence_length=4096,
)
# x is 4096.0
```

## What the gate checks

The gate constructs several configurations and computes the expected multiplier
using an independent NumPy oracle. The returned value must have relative error
$\le 10^{-9}$ compared with the oracle result.

The check does not accept a fixed table of answers. It recomputes the activation
memory equations for every tested configuration.
