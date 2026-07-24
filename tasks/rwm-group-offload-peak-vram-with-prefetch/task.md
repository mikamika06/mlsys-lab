## Context

Large models are often too big to keep fully resident on a GPU, so inference
libraries (e.g. Diffusers' `enable_group_offload`, Accelerate's sequential CPU
offload) stream weights between CPU and GPU on demand. The peak VRAM used by
weights depends heavily on *which offload strategy* is used and at *what
granularity* the model is partitioned:

**Group offloading with prefetch.** The model's leaf modules are grouped into
$G$ contiguous blocks with byte sizes $g_1, \dots, g_G$ (e.g. every $N$
transformer layers form one group). While group $i$ is executing on the GPU,
the *next* group $i+1$ is asynchronously prefetched onto the GPU so it is
ready the instant group $i$ finishes — group $i$ is only evicted once group
$i+1$'s compute begins. So at the worst moment, two groups are resident
simultaneously:

$$
\text{peak}_{\text{group}} = 2 \cdot \max_{1 \le i \le G} g_i
$$

**Sequential (leaf) offloading.** The same model is instead partitioned at
finer, leaf-module granularity — sizes $\ell_1, \dots, \ell_L$ (with
$\sum_i g_i = \sum_j \ell_j$, i.e. it's the same model, just cut into more,
smaller pieces). Each leaf is moved onto the GPU immediately before it runs
and evicted immediately after, with no prefetch overlap, so only one leaf is
ever resident:

$$
\text{peak}_{\text{sequential}} = \max_{1 \le j \le L} \ell_j
$$

**Full model offload (i.e. no offload).** The entire model stays resident on
the GPU the whole time:

$$
\text{peak}_{\text{model}} = \sum_{i=1}^{G} g_i
$$

Because leaf modules are much smaller than groups, $\text{peak}_{\text{sequential}}$
is usually far below $\text{peak}_{\text{group}}$ — group offloading trades a
higher VRAM peak for much better throughput (the prefetch overlap hides
transfer latency behind compute), while leaf-level sequential offload
minimizes VRAM at the cost of a transfer stall before almost every leaf.

## Task

Implement:

```python
def offload_peak_vram(group_sizes: np.ndarray, leaf_sizes: np.ndarray) -> dict:
    ...
```

* `group_sizes` — 1-D NumPy array of positive numbers, the byte size of each
  offload group ($g_1,\dots,g_G$).
* `leaf_sizes` — 1-D NumPy array of positive numbers, the byte size of each
  leaf module of the *same* model ($\ell_1,\dots,\ell_L$), so
  `leaf_sizes.sum() == group_sizes.sum()`.

Return a `dict` with exactly these three keys, each a plain Python `float`:

* `"group"` — peak resident bytes under group offloading with prefetch:
  $2 \cdot \max(g_i)$.
* `"sequential"` — peak resident bytes under leaf-level sequential offloading:
  $\max(\ell_j)$.
* `"model"` — peak resident bytes with no offload (whole model on device):
  $\sum_i g_i$.

## Example

```python
import numpy as np
group_sizes = np.array([4.0, 6.0, 5.0])   # 3 groups, biggest is 6
leaf_sizes  = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0])  # same model, finer cut, biggest leaf is 3

offload_peak_vram(group_sizes, leaf_sizes)
# -> {"group": 12.0, "sequential": 3.0, "model": 15.0}
```

## What the gate checks

The **exact_match** gate builds several random model partitions (random group
and leaf sizes with equal totals) and computes the reference peaks with
NumPy's `max`/`sum`. Your three returned peaks must match the reference
values exactly (bit-for-bit, since these are simple max/sum reductions with
no rounding involved) on every trial.
