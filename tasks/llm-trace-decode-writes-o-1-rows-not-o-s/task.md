## Context

During autoregressive decoding a transformer processes **one new token per step**.
Without a cache, computing attention for step $t$ would re-project and re-attend
over the whole prefix of length $t$, an $O(t)$ recompute that turns generating $S$
tokens into $O(S^2)$ work.

A **KV cache** removes the recompute. The keys and values of every past token are
already stored, so a decode step only has to *append the single new row* to each
cache and attend the new query over what is now there:

$$
K_{t} = \begin{bmatrix} K_{t-1} \\ k_t^{\top} \end{bmatrix}, \qquad
V_{t} = \begin{bmatrix} V_{t-1} \\ v_t^{\top} \end{bmatrix},
$$

$$
\mathrm{out}_t = \operatorname{softmax}\!\left(\frac{q_t\,K_t^{\top}}{\sqrt{d}}\right) V_t .
$$

The cache grows by **exactly one K row and one V row** each step — $O(1)$ writes —
never a full re-materialization of all $S$ rows. This task *measures* that
property: a correct step's Python-level work is a small constant that does **not**
grow with the cache length $S$.

## Task

Implement `decode_step`:

```python
def decode_step(k_cache, v_cache, q, k_new, v_new):
    ...
```

- `k_cache`, `v_cache`: float arrays of shape $(S, d)$ — the keys/values already
  cached for the $S$ previous tokens ($S$ may be $0$).
- `q`: float array $(d,)$ — the query for the new token.
- `k_new`, `v_new`: float arrays $(d,)$ — the key/value for the new token.

Return `(out, k_cache2, v_cache2)`:

- `out`: $(d,)$ attention output $\operatorname{softmax}(q\,K^{\top}/\sqrt{d})\,V$
  over all $S+1$ cached tokens (the new one included).
- `k_cache2`: $(S+1, d)$ cache with `k_new` appended as its **last** row.
- `v_cache2`: $(S+1, d)$ cache with `v_new` appended as its **last** row.

Append exactly one new K row and one new V row — $O(1)$ Python-level work per step,
independent of $S$. Do **not** rebuild the cache with a Python `for` loop over the
$S$ existing rows: the whole point of a KV cache is to avoid touching them again.

## Example

```python
import numpy as np
k_cache = np.array([[1.0, 0.0]])          # S = 1 cached token
v_cache = np.array([[2.0, 3.0]])
q       = np.array([0.0, 0.0])            # uniform attention (all scores 0)
k_new   = np.array([0.0, 1.0])
v_new   = np.array([4.0, 5.0])

out, k2, v2 = decode_step(k_cache, v_cache, q, k_new, v_new)
# k2 = [[1., 0.], [0., 1.]]   # new key appended as the last row
# v2 = [[2., 3.], [4., 5.]]   # new value appended as the last row
# out = [3., 4.]              # mean of the two cached values (uniform weights)
```

## What the gate checks

Two gates.

- $\mathrm{max\_abs\_err}$: the returned `out`, `k_cache2`, and `v_cache2` are
  compared against a NumPy full-attention oracle that concatenates the new row and
  attends over the whole cache. It must satisfy
  $\mathrm{max\_abs\_err} \le 10^{-9}$ across several cache lengths (including the
  empty cache $S = 0$). Because the full grown caches are checked, this also
  enforces that you appended exactly one correct row in the right place.
- $\mathrm{op\_count}$: the number of Python line events executed during **one**
  decode step at a large cache ($S = 512$), recorded with `sys.settrace`. A correct
  $O(1)$ append traces a small constant number of lines regardless of $S$; a Python
  loop that rebuilds all $S$ rows traces hundreds. The gate requires
  $\mathrm{op\_count} \le 200$, so only an $O(1)$-per-step implementation passes.
