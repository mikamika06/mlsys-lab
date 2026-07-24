## Context

"Paged" optimizers (e.g. bitsandbytes' paged AdamW) keep optimizer
state (Adam moments, etc.) in CUDA unified memory instead of pinned GPU
memory. Normally the whole working set fits on the GPU, but a transient
memory **spike** — a batch-size jump, an activation-checkpoint burst, a
big attention matrix — can momentarily need more distinct state pages
than the GPU budget allows. Instead of raising an out-of-memory error,
the runtime **pages**: it evicts the least-recently-used resident pages
to CPU to make room, and faults them back in on demand when they're
touched again. Correctness of this scheme rests entirely on doing
eviction *before* admission, every time, so the resident set never
exceeds the budget — no allocation ever fails, even under the spike.

Formally, given a budget of $B$ pages and a trace of page accesses
$p_1, p_2, \dots, p_n$, maintain a resident set $R$, $|R| \le B$ always.
On each access to page $p_t$:

$$
p_t \in R \implies \text{hit: mark } p_t \text{ most-recently-used (no change to } |R|\text{)}
$$

$$
p_t \notin R \implies \text{fault: if } |R| = B,\ \text{evict } \operatorname{LRU}(R)\ \text{first; then } R \leftarrow R \cup \{p_t\},\ p_t \text{ becomes MRU}
$$

## Task

Implement `simulate_paged_eviction`:

```python
def simulate_paged_eviction(trace: list[int], budget_pages: int) -> dict:
    ...
```

- `trace`: list of page ids (`int`) accessed in order.
- `budget_pages`: max resident pages allowed at once, a positive `int`.

Replay `trace` applying the LRU paging policy above. Return:

```python
{
  "fault_count": int,           # total faults over the whole trace
  "evicted_pages": list[int],   # page ids evicted, IN EVICTION ORDER
  "final_resident": list[int],  # pages resident at the end, LRU..MRU order
}
```

## Example

```python
trace = [0, 1, 2, 0, 1, 2, 3, 4, 5]
simulate_paged_eviction(trace, budget_pages=3)
# 0,1,2 fault in (3 faults, resident={0,1,2}, budget exactly full)
# 0,1,2 repeat -> hits, each becomes MRU in turn -> LRU order stays 0,1,2
# 3 faults in: resident full -> evict LRU (0) -> resident={1,2,3}
# 4 faults in: evict LRU (1) -> resident={2,3,4}
# 5 faults in: evict LRU (2) -> resident={3,4,5}
# -> {"fault_count": 6, "evicted_pages": [0, 1, 2], "final_resident": [3, 4, 5]}
```

## What the gate checks

The grader replays several seeded traces (hand-built and NumPy-seeded)
against the oracle LRU policy above, including one with a low-locality
working set that repeats before and after a spike of many brand-new
page ids, and a `budget_pages=1` edge case where every fault evicts the
previous page. It never calls your function to build the oracle.

`fault_count_match` requires your returned `fault_count` to equal the
oracle's on every case (must be `>= 1.0`) — this alone already catches
a wrong eviction policy on traces with revisited pages (e.g. forgetting
to mark a hit as most-recently-used turns your policy into FIFO, which
faults on pages a true LRU would have kept resident). `evicted_pages`
requires your evicted-page list, in eviction order, to equal the
oracle's exactly on every case (must be `>= 1.0`) — this additionally
catches evicting the wrong page even when the total fault count happens
to come out right.
