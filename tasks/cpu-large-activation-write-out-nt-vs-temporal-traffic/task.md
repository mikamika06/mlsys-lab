## Context

Modern CPUs have hierarchical caches and may use *non‑temporal* (streaming) stores to avoid polluting caches when writing large, one‑time outputs such as neural‑network activations or checkpoint buffers.  
A normal write of a cache line uses **read‑for‑ownership (RFO)**: the CPU first reads the old line from memory before overwriting it. Non‑temporal stores skip the RFO and may bypass the last‑level cache, reducing total DRAM traffic when the written data will not be reused soon.

Suppose an activation tensor of $N$ floats must be written linearly to memory. Each float has 4 bytes. We can model the memory traffic of two variants:

* **Temporal store (ordinary)** — each line must first be fetched (RFO) and then written back.
* **Non‑temporal store** — write combines 64‑byte lines directly to DRAM, no prefetch nor cache pollution.

A deterministic cache simulator can model which lines are brought into or evicted from cache during this write stream.

## Task

Implement

```python
def simulate_activation_write(n: int, store_type: str) -> list[int]:
    """
    Return a simulated physical write trace of byte addresses
    for writing out an activation of n float32 values.
    store_type is either "temporal" or "non-temporal".
    """
```

Your function must produce a list of *byte addresses* (e.g. `[0, 4, 8, …]`) that represent the sequence of stores issued by the CPU.  
If `store_type == "temporal"`, each store's cache line is **read first** (model an RFO read before writing the same addresses).  
If `store_type == "non-temporal"`, emit only sequential writes, representing write‑combining without a read.

All addresses must be multiples of 4. Assume each cache line holds 64 bytes, i.e. 16 float32 per line.

### Modeling assumptions

* Infinite physical memory, contiguous at address 0 onward.
* For RFO: before each first write to a line, an additional *read* over that 64‑byte region occurs (simulate by adding addresses for that read phase).
* For non‑temporal: only direct writes, no reads, same contiguous order.
* The simulator provided (``arena.cachesim.simulate``) will evaluate total cache misses of your emitted address sequence under a fixed cache geometry.

Your goal is to minimize modeled cache DRAM traffic by choosing the correct type and access pattern.

## Example

```python
from cpu_large_activation_write_out_nt_vs_temporal_traffic import simulate_activation_write

trace = simulate_activation_write(32, "non-temporal")
# writes 32 float32s -> 128 bytes (2 lines)
# expected to show no read-for-ownership prefetches
# trace[:10] => [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
```

## What the gate checks

The grader runs your `simulate_activation_write(n, type)` for several sizes (`n=1024`, `n=4096`, etc.), feeds the emitted addresses into a deterministic cache simulator using parameters `(line_bytes=64, sets=64, ways=8)`, and computes total cache *misses* including RFO fetches.  
Your result must achieve modeled cache‑miss traffic within $5\%$ of the optimal non‑temporal write baseline.  
Hardcoded answers are rejected, since the grader recomputes the reference internally.
