## Context

FlashAttention never materializes the full attention matrix: each thread
processes its query row's keys in K-BLOCKS, keeping a running max score
`m`, a running softmax denominator `l`, and a running weighted-value
accumulator `acc`. Every time a new block's own max exceeds the running
max, the OLD running statistics were computed relative to the OLD
(smaller) max — they have to be rescaled by
$\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ before the new block's
(already new-max-relative) contribution is added in:

$$
l \leftarrow l \cdot \alpha + l_{\text{block}}, \qquad
\mathrm{acc} \leftarrow \mathrm{acc} \cdot \alpha + \mathrm{acc}_{\text{block}}
$$

Both `l` and `acc` have to be rescaled by the *same* $\alpha$ — they're
two halves of the same running softmax, and the final answer is
`acc / l`. Rescale one and not the other and they silently drift out of
consistent normalization: the output isn't NaN or obviously broken, just
wrong, and only on queries where a later block actually raises the
running max.

## Task

`solve.cu` computes 8 queries' attention output over 16 keys (`D=4`),
processed as 2 blocks of 8 keys, one thread per query row — but it has
exactly this bug: `l` is correctly rescaled by `alpha` on every block
merge, but the accumulator (`acc0..acc3`) is not. Find the merge step
and fix it so the accumulator is rescaled by the same `alpha` as `l`,
**before** adding that block's own contribution — matching `l`'s line
exactly in structure.

## Example

Block 0 gives `m=2.0, l=5.0, acc=12.0` (single scalar for illustration).
Block 1's own max is `4.0`, so `new_m=4.0`,
`alpha = exp(2.0 - 4.0) ≈ 0.135`. If block 1 contributes `l_block=3.0,
acc_block=9.0`: the correct merge is `l = 5.0*0.135 + 3.0 ≈ 3.68`,
`acc = 12.0*0.135 + 9.0 ≈ 10.62`. Skipping the accumulator's rescale
gives `acc = 12.0 + 9.0 = 21.0` instead — `acc/l` comes out completely
different once you divide by the (correctly rescaled) `l`.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
a fixed input where the SECOND key block is deliberately larger in
magnitude than the first (so the running max genuinely increases on that
merge — `alpha != 1`, meaning the bug's effect can't cancel out). It
compares the kernel's output against a standard (unblocked) softmax
attention computed directly in numpy — mathematically identical to a
*correctly* merged blocked computation, so any mismatch is a real merge
bug, not a difference in algorithm. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

The shipped `solve.cu` measures `max_abs_err ≈ 1.30` — the missing
accumulator rescale is silent (no crash, no NaN, a plausible-looking
finite number) but wrong on every query where the second block changed
the running max.
