# KV cache plan for a hybrid model

Our new model is hybrid: some layers attend to the whole context, others only
to a window of the most recent tokens. The server doesn't know this and
allocates every layer a cache sized for the full context. As a result, the
box holds half as many sessions as it should, and at 8k context it OOMs
where it used to run fine at 4k.

We need to figure out how much is actually needed and put a number on it.

## What you write

`kvplan/groups.py` — `build_groups(config) -> list[group]`. The config is
`{"layers": [{"index", "kind", "window", "kv_heads", "head_dim"}, ...]}`, where
`kind` is `"full"` or `"sliding"`. Layers that need the same cache go into
one group. A group is `{"kind", "window", "kv_heads", "head_dim", "layers"}`,
where `layers` is the sorted list of indices, and `window` for full attention
is 0. Groups are ordered by their key.

`kvplan/memory.py`:

```python
group_bytes(group, max_context, block_size, bytes_per_element)
plan_bytes(config, max_context, block_size, bytes_per_element)
uniform_bytes(config, max_context, block_size, bytes_per_element)
```

Per token, a layer holds keys **and** values:
`2 · kv_heads · head_dim · bytes_per_element`. Memory is allocated in blocks,
i.e. rounded up to `block_size`. A windowed layer can't need more than the
full context would.

`kvplan/schedule.py` — `free_schedule(window, block_size, steps)`. At each
step `t` from 1 to `steps`, return **how many blocks can already be freed**:
everything that has fallen outside the window. The number can't decrease as
steps go on.

## How it's graded

The grader computes the reference itself, from the same config, across three
different models and two context sizes. The third milestone is yours: you
write a test, and we swap in a grouping implementation that merges layers
with different windows into a single group. Your test needs to catch it.

```
mlsys project start m-build-kv-cache-groups-from-a-hybrid-model-config
mlsys project grade m-build-kv-cache-groups-from-a-hybrid-model-config --milestone 1
```
