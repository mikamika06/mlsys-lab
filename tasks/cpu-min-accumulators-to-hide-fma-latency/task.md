## Context

A fused multiply-add (FMA) pipeline can execute independent operations in parallel, but
a result cannot be reused until the instruction latency has elapsed. If the FMA latency
is $L$ cycles and the processor can start $T$ FMAs per cycle, a single dependency chain
cannot saturate the pipeline.

To hide latency, independent accumulator chains are used. The minimum number of
accumulators is

$$
N = \lceil L \cdot T \rceil .
$$

Each accumulator carries its own dependency chain, allowing the processor to issue new
FMAs while previous results are still waiting.

The memory access order of a kernel also affects cache behaviour. The gate uses a
deterministic cache simulator instead of measuring real hardware. An access trace is
converted into cache-line accesses and evaluated with fixed cache parameters.

## Task

Implement `min_fma_accumulators`:

```python
def min_fma_accumulators(latency: int, throughput: float, length: int, line_bytes: int, sets: int, ways: int):
    ...
```

Return a pair:

```python
(accumulators, addresses)
```

where `accumulators` is the minimum integer number of independent FMA accumulators
needed to hide the latency, and `addresses` is a list of byte addresses representing a
contiguous cache-friendly access pattern.

The accumulator count must satisfy

$$
\mathrm{accumulators} = \lceil \mathrm{latency} \cdot \mathrm{throughput} \rceil .
$$

The address trace must contain exactly `length` addresses. The address of access $i$ is

$$
a_i = i \cdot \mathrm{line\_bytes}.
$$

The cache parameters are provided to the function because the gate uses the same
parameters when replaying the trace.

## Example

For a latency of $5$ cycles and a throughput of $0.5$ FMAs per cycle:

```python
acc, addrs = min_fma_accumulators(5, 0.5, 4, 64, 8, 2)
# acc == 3
# addrs == [0, 64, 128, 192]
```

## What the gate checks

The gate computes the accumulator reference with the latency hiding formula and checks
for an exact match.

The gate also computes the reference streaming trace itself, runs both traces through
the deterministic cache simulator, and checks that the returned trace has identical
cache miss behaviour. No wall-clock timing or real processor measurements are used.
