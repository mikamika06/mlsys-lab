## Context

A long prompt doesn't have to be prefilled in one shot: production serving
engines split it into a **schedule** of sequential chunks
$C_0, C_1, \dots, C_{T-1}$ (sizes chosen by a scheduler, not necessarily
equal) and run one forward pass per chunk, in order. Chunk $t$ covers
query positions $[s_t, e_t)$ and must produce the exact same attention
output those positions would get from a single monolithic prefill over
the whole prompt — chunking is purely a scheduling/memory trick, it must
not change the numbers.

For chunk $t$'s queries, the causally-visible keys/values are everything
at positions $[0, e_t)$: all earlier chunks' keys in full, plus chunk
$t$'s own keys up to the diagonal. Rather than materializing the whole
$e_t \times e_t$ score block at once, the chunk's output is built by
folding in one **previously-processed chunk of KV at a time** with a
running (carried) online softmax — exactly the flash-attention recurrence,
now indexed by the chunk schedule instead of a fixed tile size:

$$
m^{\text{new}} = \max(m, \max_c \text{scores}_{:,c}), \qquad
p = \exp(\text{scores} - m^{\text{new}})
$$

$$
\ell \leftarrow \ell \, e^{m - m^{\text{new}}} + \sum_c p_{:,c}, \qquad
\text{acc} \leftarrow \text{acc} \, e^{m - m^{\text{new}}} + p\,V_{\text{chunk}}, \qquad
m \leftarrow m^{\text{new}}
$$

started fresh ($m=-\infty,\ \ell=0,\ \text{acc}=0$) for each new query
chunk $t$, folding in KV chunks $u = 0, \dots, t$ in order. Every KV chunk
$u < t$ is fully visible to chunk $t$'s queries (no masking needed); KV
chunk $u = t$ is chunk $t$'s own keys, which need the usual
elementwise causal mask (query row $r$ sees key column $c$ only if
$c \le r$, both measured from the chunk's own start).

## Task

Implement `chunked_causal_prefill`:

```python
def chunked_causal_prefill(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, chunk_sizes: list[int],
) -> np.ndarray:
    ...
```

- `q, k, v`: `(n, d)` float64 arrays for the **whole** prompt.
- `chunk_sizes`: a list of positive ints summing to `n` — the prefill
  schedule; `chunk_sizes[0]` is processed first and covers positions
  `[0, chunk_sizes[0])`, `chunk_sizes[1]` covers the next span, etc.
  Sizes need not be equal.

Return the `(n, d)` causal self-attention output
$\mathrm{softmax}(QK^\top/\sqrt{d})V$ (future positions masked), computed
by processing `chunk_sizes` in order and, for each query chunk, carrying
a running `(m, l, acc)` online-softmax state across the KV chunks that
precede or equal it (as described above) rather than forming the full
dense score matrix for the whole prompt at once.

## Example

```python
import numpy as np

n, d = 10, 4
rng = np.random.default_rng(0)
q = k = v = rng.normal(size=(n, d))

out = chunked_causal_prefill(q, k, v, chunk_sizes=[3, 2, 5])
# out must equal dense causal self-attention over the full (10, 4) input,
# regardless of how the 10 positions were split into chunks -- running the
# same q, k, v through chunk_sizes=[10] or [1]*10 must give the same `out`.
```

## What the gate checks

The grader builds several `(n, d)` prompts from a seeded NumPy generator
and, for each, tests **multiple different chunk schedules** against the
same `q, k, v` (uneven sizes, a single all-in-one chunk, and many
size-1 chunks) — plus computes the reference output independently as
**single-shot dense** causal attention in float64
(`softmax(QK^T/sqrt(d)` with future positions masked`) @ V`, over the
whole `(n, n)` score matrix at once), never calling your function and
never depending on `chunk_sizes`.

`max_abs_err` is the worst-case elementwise absolute error between your
chunked output and the dense oracle, across every scenario and schedule,
and the gate requires `<= 1e-5`. Forgetting to include a chunk's own
diagonal KV block, masking the wrong triangle within it, dropping an
earlier chunk's contribution, or getting the running-max correction
factor wrong will all produce a visible mismatch for at least one
schedule.
