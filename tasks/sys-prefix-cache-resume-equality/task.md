## Context

A KV-cache lets an autoregressive decoder avoid recomputing keys/values for
tokens it has already processed. When you "resume" generation after a
prompt, only the *new* tokens' queries, keys, and values need to be
computed; the new queries then attend over the **old cached keys/values
plus the new ones**. For this to be a valid optimization, resuming from a
saved cache must produce byte-for-byte the same output as if you had run
the whole sequence (prompt + continuation) from scratch with dense causal
attention.

## Task

Implement `run_with_cache`:

```python
def run_with_cache(Wq, Wk, Wv, X, kv_cache=None):
    ...
```

- `Wq`, `Wk`, `Wv` — `(d, d)` float64 projection matrices (single head, no
  output projection).
- `X` — `(L, d)` float64, the new tokens' input embeddings to process.
- `kv_cache` — either `None` (no prefix processed yet) or a dict
  `{"K": (P, d), "V": (P, d)}` holding the already-projected keys/values of
  `P` earlier ("prefix") tokens.

Compute fresh queries/keys/values for the new tokens:
$$
Q_{\text{new}} = X W_q,\qquad K_{\text{new}} = X W_k,\qquad V_{\text{new}} = X W_v .
$$
Build the extended key/value tables by concatenating the prefix cache (if
any) with the new keys/values along the sequence axis:
$K_{\text{all}} = [K_{\text{prefix}}; K_{\text{new}}]$,
$V_{\text{all}} = [V_{\text{prefix}}; V_{\text{new}}]$.

New token $i$ (0-indexed within `X`, global position $P+i$) attends
causally over **every** global position `0..P+i` inclusive — the whole
cached prefix plus every new token up to and including itself:
$$
\text{out}_i = \operatorname{softmax}\!\left(\frac{q_i K_{\text{all}}[0{:}P{+}i{+}1]^\top}{\sqrt{d}}\right) V_{\text{all}}[0{:}P{+}i{+}1] .
$$

Return `(out, new_cache)` where `out` is `(L, d)` and
`new_cache = {"K": K_all, "V": V_all}` (shape `(P+L, d)` each). The first
`P` rows of `new_cache["K"]`/`["V"]` must be exactly the input cache's rows,
unchanged.

## Example

```python
import numpy as np
d = 4
Wq, Wk, Wv = (np.random.randn(d, d) for _ in range(3))

X1 = np.random.randn(3, d)          # the "prompt"
out1, cache1 = run_with_cache(Wq, Wk, Wv, X1, None)

X2 = np.random.randn(2, d)          # the "continuation"
out2, cache2 = run_with_cache(Wq, Wk, Wv, X2, cache1)

# out2 must equal the last 2 rows of a from-scratch dense causal-attention
# run over np.concatenate([X1, X2]) -- resuming must be indistinguishable
# from having processed the whole sequence at once.
```

## What the gate checks

* **max_abs_err** — for several seeded `(d, P, L)` configurations, the
  grader runs `run_with_cache` once on the prompt (`kv_cache=None`) and once
  on the continuation (using the returned cache), and compares both outputs
  against a dense causal-attention reference run over the full concatenated
  sequence, at the matching rows (`<= 1e-5`).
* **bookkeeping_exact** — the final `new_cache["K"]`/`["V"]` after both
  calls must have the right shape `(P+L, d)` and match a reference cache
  built the same way (`== 1.0`) — this catches an implementation that gets
  lucky on the output but corrupts, drops, or duplicates cache entries.
