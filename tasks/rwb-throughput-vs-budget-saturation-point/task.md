## Context

A server's per-iteration **token packer** fills one batch by walking the
waiting queue **in order** and greedily admitting each request whose
tokens still fit under a fixed per-iteration `budget`: it keeps a running
total and, for request $i$, admits it (adding its tokens to the running
total) only if doing so wouldn't exceed `budget`; otherwise it skips that
request (permanently, for this iteration) and moves to the next.

$$
\text{tokens\_processed}(B) = \sum_{i \,:\, \text{admitted}} \text{tokens}_i,
\qquad \text{admitted iff running\_sum} + \text{tokens}_i \le B
$$

Because relaxing $B$ only ever lets *more* requests pass the "still
fits" test, $\text{tokens\_processed}(B)$ is **non-decreasing** in $B$.
It can't grow forever, though: the running total after considering every
request in the queue is exactly $\sum_i \text{tokens}_i$, so once
$B \ge \sum_i \text{tokens}_i$ every request fits regardless of order,
and pushing $B$ any higher buys nothing more. That threshold —
$B^{*} = \sum_i \text{tokens}_i$ — is the **saturation point**: the
smallest budget beyond which per-iteration throughput stops improving.

## Task

Implement `throughput_vs_budget`:

```python
def throughput_vs_budget(
    request_tokens: list[int], budgets: list[int],
) -> tuple[list[int], int]:
    ...
```

- `request_tokens`: a list of `N` positive ints, tokens needed per
  waiting request, in the exact queue order the packer walks them.
- `budgets`: a list of candidate per-iteration budgets to evaluate
  (positive ints; order matters for the output, may contain duplicates
  or be unsorted).

Return `(throughput_curve, saturation_budget)`:

- `throughput_curve`: a list the same length as `budgets`, where entry
  `j` is $\text{tokens\_processed}(\text{budgets}[j])$ as defined above.
- `saturation_budget`: the true saturation point $B^{*}$ — the smallest
  budget at which per-iteration throughput stops increasing with more
  budget. **Not** restricted to the given `budgets` list.

## Example

```python
request_tokens = [5, 8, 2, 10]
throughput_vs_budget(request_tokens, budgets=[6, 9, 100])
# B=6:  admit 5 (sum=5); 8 doesn't fit (5+8=13>6), skip; 2 doesn't fit
#   either (5+2=7>6), skip; 10 doesn't fit -> tokens_processed(6) = 5
# B=9:  admit 5 (sum=5); 8 doesn't fit (13>9), skip; 2 fits (5+2=7<=9),
#   admit (sum=7); 10 doesn't fit (17>9) -> tokens_processed(9) = 7
# B=100: everything fits -> tokens_processed(100) = 25
# saturation_budget = sum(request_tokens) = 25
# -> ([5, 7, 25], 25)
```

## What the gate checks

The grader builds several `(request_tokens, budgets)` scenarios from a
seeded NumPy generator (queues with a mix of large and small requests in
different orders, budget lists that don't include the true saturation
point, budgets both above and below every partial sum, and duplicate or
unsorted budget lists) and computes the reference `throughput_curve` and
`saturation_budget` independently in Python by the exact greedy-packing
definition above, never calling your function or hardcoding an expected
value.

Two gates apply: `rel_err` is the worst-case relative error between your
`throughput_curve` and the oracle's, across every scenario (must be
`<= 1e-9`), and `exact_match` is the fraction of scenarios where your
`saturation_budget` equals the oracle's exactly (must be `1.0`).
Re-ordering requests (e.g. sorting by size) instead of respecting queue
order, letting a skipped request be revisited later in the same budget's
pass, or reporting the largest single request's size or the largest
partial sum as the saturation point instead of the true total will all
diverge from the oracle.
