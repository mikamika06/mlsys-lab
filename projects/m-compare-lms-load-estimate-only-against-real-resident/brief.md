# Real model memory vs. estimate-only

The local runner's panel shows, before load, how much memory a model will
need. We trusted that number and sized a fleet around it: 32 GB machines get
this model with a 16k context, 16 GB machines get the same model with 4k.

In prod it turned out differently. Machines where the estimate left a couple
of gigabytes to spare start offloading layers to the CPU, and speed drops by
a factor of three. Machines where the estimate promised a tight fit sometimes
run fine. So the number on the panel isn't lying at random — it's shifted
systematically, and we need to understand by how much and why, rather than
tacking on a "just add some headroom" fudge factor.

## What you write

`lmsmem/estimate.py` — what the panel computes:

```python
param_count(config) -> int
kv_cache_bytes(config, context_length, kv_bytes_per_element=2) -> int
estimate_bytes(config, context_length, bytes_per_param) -> int
```

Model configuration is `hidden_size`, `n_kv_heads`, `head_dim`,
`intermediate_size`, `vocab_size`, `n_layers`. The estimate is made up of the
weights plus the KV cache for the given context, the same way the panel
computes it: exactly as many bytes as the numbers ask for.

`lmsmem/resident.py` — what the process actually occupies:

```python
resident_bytes(config, context_length, bytes_per_param, page_size=4096) -> int
relative_error(estimate, resident) -> float
```

The difference isn't pulled out of thin air. Memory is handed out in pages,
and each separate region gets rounded up to a page boundary individually,
not the total size rounded up once. On top of that, the runtime keeps its
own buffers. Your job is to compute this so the gap is explained by
arithmetic, not by a "margin".

`lmsmem/offload.py` — what happens when it doesn't fit:

```python
offload_split(total_bytes, gpu_ratio, page_size=4096) -> {"gpu_bytes", "cpu_bytes"}
```

The share that stays on the accelerator is also a multiple of the page size,
and never exceeds the total size. The two parts must sum to the whole — an
invariant the grader checks separately.

## How it's graded

The grader computes the reference itself, on twelve real-model
configurations and a few context lengths. Milestone three is yours: you
write a test, and we swap `resident_bytes` so that it returns exactly the
estimate. That is, the gap disappears — the very gap this task exists for.
Your test has to catch that.

```
mlsys project start m-compare-lms-load-estimate-only-against-real-resident
mlsys project grade m-compare-lms-load-estimate-only-against-real-resident --milestone 1
```
