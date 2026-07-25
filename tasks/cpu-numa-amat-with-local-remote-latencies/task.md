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

Implement `compute_numa_amat` (declared in `sol.hpp`) which returns the
modeled AMAT (in nanoseconds) for a given NUMA configuration:

```cpp
double compute_numa_amat(double l3_hit_time_ns, double l3_miss_rate,
                          double local_dram_latency_ns, double remote_base_latency_ns,
                          int num_remote_hops, double per_hop_latency_ns,
                          double local_dram_fraction);
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

```
compute_numa_amat(
    l3_hit_time_ns=10, l3_miss_rate=0.05,
    local_dram_latency_ns=80, remote_base_latency_ns=100,
    num_remote_hops=2, per_hop_latency_ns=25,
    local_dram_fraction=0.7)
// -> 14.675
//
// remote_latency = 100 + (2-1)*25 = 125 ns
// dram_latency   = 0.7*80 + 0.3*125 = 93.5 ns
// AMAT           = 10 + 0.05*93.5 = 14.675 ns
```

## What the gate checks

`main.cpp` is a fixed driver that evaluates `compute_numa_amat` on seven
deterministic configurations (including the zero-miss-rate, all-local and
all-remote edge cases) and prints each resulting AMAT with 9 decimal
digits. The gate is `max_abs_err`: it compiles your code against the same
driver, byte-for-byte extracts the printed numbers from your binary and
from the reference binary, and requires every AMAT to match to within
$10^{-6}$ ns. Any correct algebraic implementation of the formulas above
will pass; skipping the per-hop term, swapping `local_dram_fraction` and
`1 - local_dram_fraction`, or applying `l3_miss_rate` to the wrong term
will all fail at least one of the seven cases.
