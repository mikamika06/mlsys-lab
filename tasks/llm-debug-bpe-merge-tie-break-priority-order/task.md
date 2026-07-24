## Context

Byte-pair encoding (BPE) tokenizers repeatedly merge the highest-priority adjacent token pair. Each merge rule has a rank, where a smaller rank means higher priority.

For a token sequence $x = (x_0, x_1, \dots, x_{n-1})$, adjacent pairs are

$$
(x_0, x_1), (x_1, x_2), \dots, (x_{n-2}, x_{n-1}).
$$

When several adjacent pairs are present in the merge table, the tokenizer must choose the pair with the smallest merge rank:

$$
\operatorname*{argmin}_{(a,b) \in P} \operatorname{rank}(a,b),
$$

where $P$ is the set of available adjacent pairs.

A common debugging issue is accidentally choosing the first matching pair encountered while scanning the sequence. That behavior depends on input order instead of BPE priority and can produce different tokens.

## Task

Implement `bpe_merge(tokens, ranks)`:

```python
def bpe_merge(tokens: list[str], ranks: dict[tuple[str, str], int]) -> list[str]:
    ...
```

The function performs one BPE merge step.

Find all adjacent pairs in `tokens` that exist in `ranks`. Select the pair with the lowest rank value. If multiple occurrences of the same pair exist, merge every occurrence in a single left-to-right pass. If no pair exists in `ranks`, return `tokens` unchanged.

When merging a pair `(a, b)`, replace adjacent occurrences with the concatenated token `a + b`.

The input list should not be modified in place.

## Example

```python
tokens = ["l", "o", "w", "e", "r"]
ranks = {
    ("e", "r"): 0,
    ("l", "o"): 5,
    ("o", "w"): 3,
}

result = bpe_merge(tokens, ranks)
# ["l", "o", "w", "er"]
```

The pair `("e", "r")` is selected because rank $0$ is the highest priority.

## What the gate checks

The gate compares the submitted implementation against an independently computed BPE reference implementation. Cases include conflicting adjacent pairs where a first-seen scan gives a different result from the lowest-rank priority rule.

The `exact_match` score must be $1.0$ for all tested cases.
