## Context

Flash-attention-style kernels never hold a whole row of attention scores in
memory at once — they stream through it in chunks, and after each chunk
they only keep two numbers: the running max seen so far, $m$, and the
running sum of exponentials shifted by that max, $l = \sum \exp(\text{score}
- m)$ (the "safe softmax" trick: shifting by the max keeps every exponent
$\le 0$, so nothing overflows). The catch is that each chunk's $l$ was
computed relative to *that chunk's own* max — chunk 1's $l_1$ used $m_1$,
chunk 2's $l_2$ used $m_2$, and $m_1 \ne m_2$ in general. You cannot just
add $l_1 + l_2$; the two sums are scaled differently.

Merging them into the statistics for the whole sequence needs one more
shift, onto the new combined max:

$$
m = \max(m_1, m_2) \qquad\qquad
l = l_1 \, e^{\,m_1 - m} + l_2 \, e^{\,m_2 - m}
$$

Each term's exponent, $m_1 - m$ or $m_2 - m$, is always $\le 0$ (since $m$
is the max of the two), so this never overflows either — and whichever
chunk's max turns out to be the new global max has its own rescale factor
equal to $e^0 = 1$, contributing its original $l_k$ unchanged. This merge
is **associative**: repeat it over any number of chunks, in any grouping,
and you get exactly the statistics a single pass over the whole sequence
at once would have produced.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void merge_online_softmax(float* m_out, float* l_out,
                                      const float* m1, const float* l1,
                                      const float* m2, const float* l2, int n);
```

For each row `i` in `[0, n)`, given only `(m1[i], l1[i])` and
`(m2[i], l2[i])` — never the raw scores — compute
`m_out[i] = max(m1[i], m2[i])` and
`l_out[i] = l1[i]*exp(m1[i] - m_out[i]) + l2[i]*exp(m2[i] - m_out[i])`.

## Example

The grader gives each of 64 rows two chunks of scores at very different
scales — about half the rows have chunk 1's values around 40 units larger
than chunk 2's, and half the reverse — so the correct merge has to handle
the new global max coming from *either* chunk. A correct merge matches the
statistics computed directly from the full concatenated row of scores
exactly (`max_abs_err = 0.0`); the starter (empty body, both outputs left
at `0.0`) misses by about `45` in the worst row — roughly the scale gap
itself, since a totally wrong `l` reported as `0` instead of a real sum is
exactly the size of error the max-scale offset was designed to expose.

## What the gate checks

`check.py` builds the two-chunk fixture, parses `solve.cu`, and runs
`merge_online_softmax` on the software GPU (`arena.cuda_sim.GPU`) with a
1-block, 64-thread launch. It requires `max_abs_err <= 1e-6` (both
`m_out` and `l_out` together) against statistics computed independently
from the raw, concatenated scores. Forgetting to rescale before adding
(`l_out[i] = l1[i] + l2[i]`) matches on rows where `m1[i] == m2[i]` by
coincidence but is wrong by a large, scale-dependent amount on every row
where the two chunks' maxima differ — which, by construction, is most of
them.
