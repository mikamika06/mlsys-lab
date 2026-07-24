## Context

Memory-history snapshots represent allocator state as segments containing blocks.
A segment contributes to reserved memory whenever it exists, while allocated memory
depends on which blocks are currently active.

For a snapshot state, let $S$ be the set of segments and $B_s$ be the blocks in
segment $s$. The reserved bytes are

$$
R = \sum_{s \in S} \mathrm{size}(s),
$$

and the allocated bytes are

$$
A = \sum_{s \in S}\sum_{b \in B_s} \mathrm{size}(b)\,\mathbf{1}[\mathrm{active}(b)].
$$

The largest single allocation is

$$
M = \max_{b \in B_s,\ \mathrm{active}(b)} \mathrm{size}(b).
$$

A history snapshot stores allocation and release events. Recovering the peaks
requires replaying those events in order while tracking the current allocator
state.

## Task

Implement `recover_memory_peaks(snapshot)`:

```python
def recover_memory_peaks(snapshot):
    ...
```

The input is a dictionary with this structure:

```python
{
    "segments": [
        {
            "id": "segment-name",
            "size": 1024,
            "blocks": [
                {"id": "block-name", "size": 128, "events": ["alloc", "free", "alloc"]}
            ]
        }
    ]
}
```

Each block starts inactive. Its `events` list is replayed from left to right.
An `"alloc"` event activates the block and a `"free"` event releases it. The same
block may be allocated multiple times after being freed.

Return a tuple:

```python
(peak_reserved, peak_allocated, largest_allocation)
```

where:

- `peak_reserved` is the maximum total segment size seen while replaying.
- `peak_allocated` is the maximum active block bytes seen while replaying.
- `largest_allocation` is the largest size of any block when an allocation event
  occurs.

Assume event sequences are valid and do not free inactive blocks.

## Example

```python
snapshot = {
    "segments": [
        {
            "id": "s0",
            "size": 1000,
            "blocks": [
                {"id": "a", "size": 200, "events": ["alloc", "free"]},
                {"id": "b", "size": 300, "events": ["alloc"]}
            ]
        }
    ]
}

recover_memory_peaks(snapshot)
# (1000, 500, 300)
```

## What the gate checks

The gate creates several memory-history snapshots and computes the expected
values by replaying the same snapshot model with an independent oracle.

The returned tuple must exactly match the oracle output for every snapshot.
