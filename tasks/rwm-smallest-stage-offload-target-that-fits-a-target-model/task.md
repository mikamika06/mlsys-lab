## Context

Large language model training memory is often reduced with ZeRO-style sharding. A model with
$P$ parameters has several memory components. This task uses the following simplified
training memory model:

$$
\text{parameters}=2P,\qquad
\text{gradients}=2P,\qquad
\text{optimizer}=8P
$$

where bytes are used and parameters are stored in fp16 while optimizer states are stored
in fp32.

For $N$ GPUs, a ZeRO stage changes which components are partitioned:

- Stage 0 keeps parameters, gradients, and optimizer states replicated.
- Stage 1 partitions optimizer states.
- Stage 2 partitions optimizer states and gradients.
- Stage 3 partitions parameters, gradients, and optimizer states.

CPU offload moves optimizer states to CPU memory. The GPU memory target is only concerned
with the remaining GPU-resident bytes.

The selected configuration should minimize overhead. The search order is:

$$
(0,\text{none}),
(1,\text{none}),
(2,\text{none}),
(3,\text{none}),
(1,\text{cpu}),
(2,\text{cpu}),
(3,\text{cpu})
$$

The first configuration in this order whose estimated per-GPU memory fits the target is
the desired result.

## Task

Implement `choose_stage_offload(Phi, N, gpu_memory)`:

```python
def choose_stage_offload(Phi: int, N: int, gpu_memory: int) -> tuple:
    ...
```

Return a tuple `(stage, offload)` where `stage` is an integer from `0` to `3` and
`offload` is either `"none"` or `"cpu"`.

The function must evaluate the ZeRO memory candidates in the specified order and return
the first configuration satisfying the per-GPU memory constraint.

## Example

```python
config = choose_stage_offload(1000000000, 8, 3000000000)
# returns something like:
# (2, "none")
```

The exact result depends on the memory limit because the function must compute the
candidate sizes.

## What the gate checks

The grader builds a reference by independently evaluating the memory equations for every
candidate configuration in order. Random model sizes, GPU counts, and memory limits are
used. The returned tuple must exactly match the first fitting configuration.
