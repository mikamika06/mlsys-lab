## Context

Production attention and distributed query systems often split a workload into multiple independent pieces. A larger split count $S$ creates more parallel tiles, but it also increases the amount of intermediate data that must be combined.

For a batch with $B$ sequences and $Q$ query rows, the number of parallel tiles is modeled as

$$P(S) = B \times Q \times S .$$

The combine stage introduces a penalty proportional to the number of splits and the key/value length:

$$C(S) = S \times K,$$

where $K$ is the key/value length. The effective utility of a split count is therefore

$$U(S) = P(S) - C(S) = BQS - SK.$$

A runtime planner can evaluate several candidate split counts and choose the one with the highest utility.

## Task

Implement `choose_split_count`:

```python
def choose_split_count(batch: int, query_rows: int, kv_len: int, split_counts: list[int]) -> tuple[list[float], int]:
    ...
```

Return a tuple containing:

1. A list of effective utility values $U(S)$ for every value in `split_counts`, preserving input order.
2. The split count $S$ with the highest utility.

Use the model above directly. If multiple split counts have the same maximum utility, return the first one in `split_counts`.

## Example

```python
values, best = choose_split_count(2, 8, 10, [1, 2, 4])

# values:
# [6.0, 12.0, 24.0]
# best:
# 4
```

## What the gate checks

The gate recomputes the utility model independently for several inputs and compares both the returned utility list and selected split count.

The returned split count and work array must exactly match the oracle output.
