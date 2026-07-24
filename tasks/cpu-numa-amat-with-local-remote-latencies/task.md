## Context

In a **Non-Uniform Memory Access** (NUMA) system each CPU socket has its own
local DRAM, and remote DRAM (attached to a different socket) is reachable through
the interconnect but at higher latency.  A common model for remote latency is a
base cost plus a per-hop additive term:

$$t_{\text{remote}} \;=\; t_{\text{base}} \;+\; (h-1)\,\cdot\,t_{\text{hop}}$$

where $h$ is the number of hops to the remote node, $t_{\text{base}}$ is the
base remote access latency, and $t_{\text{hop}}$ is the additional latency per
extra hop.

The **Average Memory Access Time** (AMAT) for a single-level cache backed by
NUMA-aware DRAM is:

$$\text{AMAT} \;=\; t_{\text{cache}} \;+\; r \;\times\; \Bigl(f \cdot t_{\text{local}} \;+\; (1 - f) \cdot t_{\text{remote}}\Bigr)$$

where $t_{\text{cache}}$ is the cache-hit time, $r$ is the cache miss rate, and
$f$ is the fraction of DRAM accesses served by the local NUMA node.

## Task

Implement `compute_numa_amat` which returns the modeled AMAT (in nanoseconds)
for a given NUMA configuration.

```python
def compute_numa_amat(
    l3_hit_time_ns: float,
    l3_miss_rate: float,
    local_dram_latency_ns: float,
    remote_base_latency_ns: float,
    num_remote_hops: int,
    per_hop_latency_ns: float,
    local_dram_fraction: float,
) -> float:
    ...
```

| Parameter | Meaning |
|---|---|
| `l3_hit_time_ns` | L3 cache hit latency ($t_{\text{cache}}$) |
| `l3_miss_rate` | Fraction of accesses that miss L3 ($r$) |
| `local_dram_latency_ns` | DRAM latency for the local NUMA node ($t_{\text{local}}$) |
| `remote_base_latency_ns` | Base latency for a remote NUMA access ($t_{\text{base}}$) |
| `num_remote_hops` | Hops to the remote node ($h$); $\ge 1$ |
| `per_hop_latency_ns` | Additional latency per extra hop ($t_{\text{hop}}$) |
| `local_dram_fraction` | Fraction of DRAM accesses hitting local node ($f$); in $[0,1]$ |

## Example

```python
compute_numa_amat(
    l3_hit_time_ns=10,
    l3_miss_rate=0.05,
    local_dram_latency_ns=80,
    remote_base_latency_ns=100,
    num_remote_hops=2,
    per_hop_latency_ns=25,
    local_dram_fraction=0.7,
)
# -> 14.675
#
# remote_latency = 100 + (2-1)*25 = 125 ns
# dram_latency   = 0.7*80 + 0.3*125 = 93.5 ns
# AMAT           = 10 + 0.05*93.5 = 14.675 ns
```

## What the gate checks

The grader evaluates your function on several parameter sets and computes the
reference AMAT using the same closed-form formula.  The gate passes when the
maximum **relative error** across all test cases satisfies:

$$\frac{|\text{AMAT}_{\text{yours}} - \text{AMAT}_{\text{ref}}|}{|\text{AMAT}_{\text{ref}}|} \;\le\; 10^{-9}$$

Any correct algebraic implementation of the formulas above will pass.
