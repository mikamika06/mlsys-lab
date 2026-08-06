## Context

During language model generation, a sampler filters the raw probability distribution to prevent degenerate text. Tokens that are filtered out are set to $-\infty$ in the logit vector before the final softmax and sampling step.

Common filtering strategies include:
- **Greedy**: Keeps only the single most likely token (arg-max). All others are $-\infty$.
- **Top-K**: Keeps exactly the $k$ most likely tokens.
- **Top-P (Nucleus)**: Sorts tokens by descending probability and keeps the smallest set of top tokens whose cumulative probability equals or exceeds $p$. Thus, the sum of probabilities of the kept tokens is $\ge p$, but the sum without the smallest kept token is $< p$.
- **Min-P**: Keeps tokens whose probability is at least $p \times P_{\max}$, where $P_{\max}$ is the probability of the most likely token.

## Task

Write `classify_sampler(orig_logits, filtered_logits)`:

```python

def classify_sampler(orig_logits: list[list[float]], filtered_logits: list[list[float]]) -> str:
    ...
```

Given a batch of original logits and the corresponding post-filter logits (where rejected tokens have been set to `-float('inf')`), classify which strategy produced the filter. Return one of `"greedy"`, `"top-k"`, `"top-p"`, or `"min-p"`.

The logits are 2D float list of shape `(batch_size, vocab_size)`. You can assume:

- The batch size is large enough (e.g., 32) to eliminate ambiguity (so top-p won't accidentally keep exactly $k$ tokens for every row).
- The sampler parameter ($k$, $p$, or $\text{min\_}p$) is constant across the entire batch.
- Floating-point arithmetic may introduce tiny inaccuracies, so use a tolerance (e.g., `1e-5`) when checking strict inequalities (like establishing upper and lower bounds for $p$).


*Hint: For Top-P and Min-P, calculate the probability distributions using softmax. Then compute the bounds on the parameter $p$ required for each row. If the intersection of valid $p$ intervals across all rows is non-empty, you've found the correct sampler.*

## Example

```python

# Pseudo-code example
classify_sampler(logits, filtered_logits_from_top_p)
# Returns: "top-p"
```

## What the gate checks

The grader applies each of the 4 samplers with randomized parameters to a batch of random logits, ensuring that no single parameter choice could accidentally overlap multiple sampling strategies. It tests whether your function correctly labels the output of each sampler as `"greedy"`, `"top-k"`, `"top-p"`, or `"min-p"`.
