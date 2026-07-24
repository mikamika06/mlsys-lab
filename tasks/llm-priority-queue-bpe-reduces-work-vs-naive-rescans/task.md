## Context

A trained byte-pair-encoding (BPE) tokenizer encodes text by repeatedly merging
adjacent symbols according to a *merge table*. The base alphabet is the byte
values $0 \dots 255$. The merge table assigns each learned merge an integer
**rank** $r = 0, 1, 2, \dots$ (the order in which it was learned): a lower rank
means higher priority. Merging the adjacent pair $(a, b)$ whose rank is $r$
replaces the two symbols with a single new symbol whose id is

$$\mathrm{id}(a, b) \;=\; 256 + r .$$

Because a merge that *produces* symbol $256 + r$ is learned at step $r$, any merge
that later *consumes* that symbol must have a strictly larger rank. This BPE
invariant means the merges are always applied in ascending rank order.

Given a sequence and the merge table, encoding is defined round by round:

1. Among all adjacent pairs currently in the sequence, find the one whose rank is
   smallest (highest priority). If no adjacent pair has a rank, stop.
2. Replace every non-overlapping occurrence of that pair, scanning left to right,
   with its merged id $256 + r$.
3. Repeat.

The **naive** implementation rescans the whole sequence on every round to find the
next best pair and to rebuild the sequence, costing $O(n)$ work per merge and
$O(n \cdot m)$ overall for $m$ merges. A **priority queue** keyed by rank fixes
this: after a merge only the two new neighbouring pairs can change, so each merge
does $O(\log n)$ work and the total drops toward $O(n \log n)$ — the same output
with far fewer symbol comparisons.

## Task

Implement `bpe_encode(ids, ranks)`:

```python
def bpe_encode(ids: list[int], ranks: dict[tuple[int, int], int]) -> list[int]:
    ...
```

- `ids` is the initial sequence of integer symbol ids (base bytes plus any already
  merged ids).
- `ranks` maps a pair of current symbol ids `(a, b)` to its integer rank.
  Merging that pair yields the id `256 + ranks[(a, b)]`.

Return the fully encoded sequence of ids. The output must match the round-based
definition above exactly. Do **not** mutate the caller's `ids` or `ranks`. Use a
priority queue (heap) so the work does not scale with (sequence length $\times$
number of merges).

## Example

```python
# alphabet: 'a'=97, 'b'=98, 'c'=99
ids   = [97, 98, 97, 98, 99]          # a b a b c
ranks = {(97, 98): 0, (256, 99): 1}   # merge 0: (a,b)->256 ; merge 1: (ab,c)->257

bpe_encode(ids, ranks)
# round 1 (rank 0): (a,b) -> 256  =>  [256, 256, 99]
# round 2 (rank 1): (256,99) -> 257 => [256, 257]
# returns [256, 257]
```

## What the gate checks

Two gates.

- `exact_match` — the grader builds several sequences and derives a valid merge
  table for each by running an independent greedy BPE trainer, then computes the
  expected encoding with a straightforward round-based reference. Your output must
  equal the reference on every case (including empty / single-symbol / overlapping
  edge cases). $1.0$ means every case matched.
- `op_count` — the number of Python line events executed by your function
  (recorded with `sys.settrace`) on the largest cases must stay at or below the
  budget. A full-rescan-per-merge solution emits tens of thousands of line events
  and blows past it; only an incremental priority-queue solution stays under.
