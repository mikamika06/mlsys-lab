## Context

During inference, a transformer can either recompute attention-related values or load
stored KV data from CPU memory. Recompute usually grows quadratically with sequence
length, while transferring KV data grows linearly.

For a sequence length $s$, model the recompute cost as

$$C_\mathrm{recompute}(s) = f s^2,$$

where $f$ is the recompute cost coefficient measured in FLOP-equivalent units per
token pair. Model CPU transfer cost as

$$C_\mathrm{load}(s) = \frac{b s}{r},$$

where $b$ is the KV bytes transferred per token and $r$ is the CPU transfer bandwidth
in bytes per second.

The crossover is the smallest integer sequence length where loading is no more
expensive than recomputing:

$$s^{*} = \min \{s \ge 1 : C_\mathrm{load}(s) \le C_\mathrm{recompute}(s)\}.$$

A production runtime can use this value to decide whether to recompute or fetch KV
state.

## Task

Implement `crossover_seq_len`:

```python
def crossover_seq_len(
    recompute_coeff: float,
    kv_bytes_per_token: float,
    bandwidth_bytes_per_s: float,
) -> int:
    ...
```

Return the smallest positive integer sequence length $s$ satisfying the crossover
condition. Use the cost model directly. The inputs are positive finite numbers.

## Example

```python
s = crossover_seq_len(
    recompute_coeff=2.0,
    kv_bytes_per_token=1000.0,
    bandwidth_bytes_per_s=1000.0,
)
# recompute cost is 2*s^2 and load cost is 1*s
# the crossover happens at sequence length 1
```

## What the gate checks

The gate builds an independent NumPy-based oracle for the two cost curves and finds
the first integer index where load cost is less than or equal to recompute cost.
Your returned `int` must match the oracle crossover index exactly.
