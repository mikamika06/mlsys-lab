## Context

The **roofline model** relates attainable performance to arithmetic intensity.
For a processor with peak compute throughput $P_\text{peak}$ (in FLOP/s) and
sustained memory bandwidth $B_\text{mem}$ (in bytes/s), the achievable
performance $P$ is bounded by

$$
P \le \min\bigl(P_\text{peak},\; B_\text{mem} \times I\bigr),
$$

where $I$ is the arithmetic intensity (FLOPs per byte transferred).  
If $I < P_\text{peak}/B_\text{mem}$, the kernel is **memory‑bound**; otherwise it
is **compute‑bound**.

In a transformer layer, the *prefill* phase (large batch, long sequence) and the
*decode* phase (batch 1, short sequence) stress the CPU differently. Prefill
streams large matrices through memory, while decode reuses small weights many
times.

## Task

Implement

```python
def roofline_phase_classify(batch_sizes, seq_lengths):
    """
    Given lists of batch sizes and sequence lengths, return a list of strings
    'memory-bound' or 'compute-bound' for each (batch, seq) pair.
    """
```

Assume a fixed CPU with

- peak compute throughput `P_peak = 200e9` FLOP/s,
- sustained memory bandwidth `B_mem = 50e9` bytes/s.

Approximate arithmetic intensity for a transformer layer as

$$
I = \frac{2\,d_\text{model}\,d_\text{ff}}{4\,d_\text{model} + 4\,d_\text{ff}}
\times \frac{\text{seq}}{\text{seq} + 64}
\times \frac{\text{batch}}{\text{batch} + 8}
\times \frac{1}{4}
$$

with $d_\text{model}=4096$ and $d_\text{ff}=11008$.  
Classify each configuration as `"memory-bound"` if
$B_\text{mem} \times I < P_\text{peak}$, else `"compute-bound"`.

Return a list of equal length to the inputs.

## Example

```python
batches = [1, 16]
seqs = [8, 512]
roofline_phase_classify(batches, seqs)
# ['memory-bound', 'compute-bound']
```

## What the gate checks

The grader recomputes the same formula and compares your list to the reference
exactly (`exact_match == 1.0`).  No timing or hardware queries are used; the
roofline is purely analytical and deterministic.
