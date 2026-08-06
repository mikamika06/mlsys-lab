## Context

Full attention forms the entire score matrix at once:

$$
O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

That matrix is $n \times n$. Ring attention exists because at long context you
cannot hold it — or even all of $K$ and $V$ — on one device. Instead each rank
owns one key/value block, the blocks rotate around the ring, and every rank
folds each arriving block into a **running state** and then drops the block.

So the property that makes ring attention work is not that it produces the right
number. It is that it produces the right number while the state stays the same
size no matter how many blocks arrive. A version that stores the blocks as they
pass and runs full attention at the end returns the same matrix and misses the
entire point.

Two things break a naive accumulator:

- summing $\sum_j e^{s_j}$ across blocks overflows once scores get large, and
- the normalisation from earlier blocks is wrong as soon as a later block
  contains a bigger score.

Both are fixed by carrying a running maximum alongside the running sum, and
rescaling what you already have whenever that maximum moves.

## Task

Implement two functions.

```python
def ring_step(state, Q, K_block, V_block, scale):
    ...

def ring_output(state) -> np.ndarray:
    ...
```

`ring_step` folds one arriving key/value block into the state and returns the
new state. It is called once per block, in ring order, and receives:

- `state` — whatever you returned last time; `None` on the first call.
- `Q` — the query matrix, shape $(n, d)$, the same object every call.
- `K_block`, `V_block` — one block, shapes $(b, d)$ and $(b, d_v)$. Block sizes
  are not necessarily equal.
- `scale` — the factor the logits must be multiplied by, i.e. $1/\sqrt{d}$.

`ring_output` turns the final state into the attention output, shape
$(n, d_v)$, dtype `float64`.

You never see more than one block per call, and the driver does not keep them
for you. Do not store `Q`, `K_block`, or `V_block` in the state, and do not
accumulate anything in module-level variables.

## Example

The driver runs this loop:

```python
state = None
for K_block, V_block in blocks:
    state = ring_step(state, Q, K_block, V_block, scale)
O = ring_output(state)
```

With one single block covering the whole sequence, the result is plain
attention. With eight blocks it must be the same matrix.

## What the gate checks

`max_abs_err` — the output against an independent NumPy full-attention oracle,
over several shapes, several block counts including uneven splits, and one case
whose logits reach roughly $700$, where an accumulator without a running maximum
overflows to `inf` and then `nan`. Must be below $10^{-9}$.

Every case is run twice from a fresh `state`. If the two runs disagree, the
implementation is keeping something between calls and both metrics fail.

`state_bytes_ratio` — the largest state observed during the loop, divided by the
$n(d_v + 2) \cdot 8$ bytes that a running $(m, l, o)$ needs in float64:

$$
\text{state\_bytes\_ratio} =
\frac{\max_{\text{steps}} \text{bytes}(\text{state})}{8\,n\,(d_v+2)} .
$$

It must be at most $1.25$. The cases use $d \gg d_v$, so a state that keeps the
blocks — or keeps `Q` — lands above $6.0$ and fails even though its
`max_abs_err` is perfect. That gate is the task.
