## Context

**Prompt-lookup decoding** is a speculative-decoding proposer that needs no
draft model: it bets that if the last few generated tokens have appeared
together earlier in the same context, whatever followed them last time is a
good guess for what comes next. Every step it searches the context already
generated for the longest earlier repeat of the current suffix and proposes
the tokens that followed that repeat, for the real (target) model to verify
in parallel.

Formally, given context $c = (c_0, \dots, c_{n-1})$ and bounds
$L_{\min} \le L_{\max}$, try suffix lengths $L = L_{\max}, L_{\max}-1,
\dots, L_{\min}$ in that order. For each $L$, the current suffix is
$c_{n-L:n}$. Search for indices $i$ with

$$
0 \le i \le n - 2L, \qquad c_{i:i+L} = c_{n-L:n},
$$

(the $i + L \le n - L$ bound keeps the earlier occurrence from overlapping
the suffix itself). Among all such $i$, take the **largest** (the most
recent match). The first $L$ (largest first) with any match wins; the
proposal is the $k$ tokens that followed that occurrence,
$c_{i+L : i+L+k}$ (fewer than $k$ if the context doesn't extend that far).
If no $L \in [L_{\min}, L_{\max}]$ has any match, the proposal is empty.

## Task

Implement `propose_tokens`:

```python
def propose_tokens(context: list[int], prompt_lookup_min: int, prompt_lookup_max: int, num_speculative_tokens: int) -> list[int]:
    ...
```

- `context` — 1-D integer array, the tokens generated/seen so far.
- `prompt_lookup_min`, `prompt_lookup_max` — inclusive bounds on the
  n-gram length to search for ($L_{\min} \le L_{\max}$).
- `num_speculative_tokens` — $k$, the maximum number of tokens to propose.

Return an integer array of the proposed tokens (length between $0$ and
$k$), per the longest-match, most-recent-occurrence rule above.

## Example

```python
context = [5, 1, 2, 3, 9, 9, 1, 2, 3]
propose_tokens(context, prompt_lookup_min=2, prompt_lookup_max=4, num_speculative_tokens=2)
# suffix [1,2,3] (L=3) matches context[1:4]; that occurrence is followed by
# [9, 9] -> proposal = [9, 9]
```

## What the gate checks

The grader takes one fixed token stream (with several deliberately repeated
sub-sequences, at different lengths, mixed with random filler so
no-match positions occur too) and calls your proposer at multiple context
lengths and multiple $(L_{\min}, L_{\max}, k)$ settings. It compares your
proposal against a brute-force reference search, element-for-element:

$$
\text{your proposal} = \text{oracle proposal} \quad \text{(exact\_match == 1.0)}.
$$

Searching only the shortest n-gram length, taking the first match instead
of the most recent one, allowing the match to overlap the query suffix, or
forgetting the empty-proposal case will disagree with the oracle on at
least one context length.
