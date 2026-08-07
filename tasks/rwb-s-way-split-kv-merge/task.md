## Context

Split-KV (a.k.a. flash-decoding) attention divides the key/value sequence
into `S` independent chunks, computed **in parallel**, each producing a
local, incomplete softmax: its own running max $m_s$, its own softmax
denominator $l_s$, and its own weighted-value accumulation $\text{acc}_s$
— none of which is a valid attention output on its own, since none of
them has seen the other chunks' scores. A separate **merge** step
combines all `S` partials into the single correct result, using the same
numerically-stable rescale trick as online softmax:

$$
m^{*} = \max_s m_s
$$

$$
\ell^{*} = \sum_{s=1}^{S} \ell_s \cdot e^{m_s - m^{*}}, \qquad
\text{acc}^{*} = \sum_{s=1}^{S} \text{acc}_s \cdot e^{m_s - m^{*}}
$$

$$
\text{out} = \text{acc}^{*} \,/\, \ell^{*}
$$

This is exactly equivalent to running attention over the full
concatenated $K, V$ at once — splitting the work into `S` chunks and
merging afterward changes nothing about the math, only how (and where)
the compute happens.

## Task

Implement `merge_split_kv`:

```python
def merge_split_kv(partials):
    ...
```

- `partials`: a list of `S` `(m_s, l_s, acc_s)` tuples, one per KV chunk,
  all for the **same** batch of query rows:
  - `m_s`: shape `(n,)` — that chunk's local running max score per query row.
  - `l_s`: shape `(n,)` — that chunk's local softmax denominator per query row.
  - `acc_s`: shape `(n, d)` — that chunk's local
    $\sum_j \exp(\text{score}_{ij} - m_{s,i}) \cdot V_j$ accumulation.

Return the `(n, d)` merged attention output, per the formulas above.

## Example

```python

# 2 chunks, n=3 query rows, d=4
m1 = [1.0, 2.0, 0.5]; l1 = [3.0, 1.0, 2.0]
acc1 = [[1.0] * 4 for _ in range(3)]
m2 = [0.0, 3.0, 0.5]; l2 = [2.0, 4.0, 1.0]
[[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]

out = merge_split_kv([(m1, l1, acc1), (m2, l2, acc2)])
# row 0: m* = max(1.0, 0.0) = 1.0
#   l* = 3.0*exp(0) + 2.0*exp(-1.0) = 3.0 + 0.7358 = 3.7358
#   acc* = 1*ones(4)*exp(0) + 2*ones(4)*exp(-1.0) -> out[0] = acc*/l*
```

## What the gate checks

The grader builds several `(q, k, v, S)` scenarios from a seeded Python
generator (varying sequence length, head dim, and number of chunks `S`,
including chunk counts that don't divide the sequence length evenly) and
computes, independently in float64: the reference full attention output
via ordinary dense `softmax(QK^T/sqrt(d))V` over the whole `k, v`, *and*
the `S` local partials `(m_s, l_s, acc_s)` by running the same local
(unmerged) softmax math over each of `S` contiguous KV chunks — then
passes those partials to your `merge_split_kv`, never calling your
function to produce anything other than the merge itself and never
hardcoding an expected output.

`max_abs_err` is the worst-case elementwise absolute error between your
merged output and the dense oracle, across every scenario, and the gate
requires `<= 1e-5`. Averaging the `acc_s` values without the
$e^{m_s - m^{*}}$ rescale, applying the rescale to `acc_s` but forgetting
it for `l_s` (or vice versa), or using the wrong sign in the exponent
will all diverge sharply whenever chunks have different local maxima.
