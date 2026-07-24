## Context

The roofline model classifies a workload by comparing its arithmetic intensity with a machine balance point.

For a transformer phase configuration with batch size $B$, sequence length $S$, and hidden dimension $d$, estimate the amount of computation as

$$
\mathrm{FLOPs} = 2BS^2d + 2BSd^2 .
$$

Estimate the memory traffic as

$$
\mathrm{Bytes} = 4(BSd + Bd^2).
$$

The arithmetic intensity is

$$
AI = \frac{\mathrm{FLOPs}}{\mathrm{Bytes}} .
$$

A machine's ridge point is the ratio between its peak compute throughput and memory throughput:

$$
R = \frac{\mathrm{Peak\ FLOPs/s}}{\mathrm{Peak\ Bytes/s}} .
$$

The roofline classification rule is:

$$
AI < R \Rightarrow \text{bandwidth-bound}
$$

and

$$
AI \ge R \Rightarrow \text{compute-bound}.
$$

The same classifier applies to prefill, decode, and chunked prefill configurations because the phase is represented by its numerical shape.

## Task

Implement `classify_roofline_region(configs)`:

```python
def classify_roofline_region(configs):
    ...
```

The input is a list of tuples:

```python
(batch, seq, d, machine_balance)
```

where the values describe a phase configuration and the machine ridge point.

Return a list of strings with the same length as `configs`. Each result must be one of:

```text
"bandwidth-bound"
```

or

```text
"compute-bound"
```

Compute arithmetic intensity using the formulas above and classify each configuration using the ridge comparison rule.

## Example

```python
configs = [
    (1, 2048, 4096, 80.0),
    (1, 32, 4096, 80.0),
]

result = classify_roofline_region(configs)

# [
#   "compute-bound",
#   "bandwidth-bound",
# ]
```

## What the gate checks

The gate creates multiple prefill, decode, and chunked prefill-like configurations. It computes the expected result with a NumPy-based oracle that independently evaluates the arithmetic intensity equations.

The returned list must exactly match the oracle output. Heuristics based only on sequence length, batch size, or hidden dimension do not pass.
