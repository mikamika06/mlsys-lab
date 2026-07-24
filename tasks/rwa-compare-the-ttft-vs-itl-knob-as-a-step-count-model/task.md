## Context

Autoregressive inference systems often split a long prompt into prefill chunks. A
larger chunk reduces the number of prefill iterations needed before the first
token, while increasing the amount of work handled in each iteration.

For a prompt length $L$ and chunk size $c$, the number of prefill steps is

$$
S(c) = \left\lceil \frac{L}{c} \right\rceil .
$$

A simple model for the batched-token load of each prefill step is to process
chunks of size $c$ until the final remainder. The per-step loads are therefore

$$
B(c) = [c, c, \dots, c, r],
$$

where $r = L \bmod c$ and the final element is omitted when $r = 0$.

This model separates two effects of the chunk-size knob: time-to-first-token
(TTFT) is approximated by the number of prefill steps, while inter-token latency
(ITL) pressure is approximated by the token load handled by each step.

## Task

Implement `compare_chunk_knob(L, chunk_sizes)`:

```python
def compare_chunk_knob(L: int, chunk_sizes: list[int]) -> list[dict]:
    ...
```

Return one dictionary for each chunk size in the input order. Each dictionary
must contain:

- `"chunk_size"`: the input chunk size $c$
- `"prefill_steps"`: $S(c)$
- `"step_token_loads"`: the list $B(c)$

Assume $L > 0$ and all chunk sizes are positive integers.

Do not sort the input. The output must preserve the exact order of
`chunk_sizes`.

## Example

```python
result = compare_chunk_knob(10, [4, 6, 10])
```

The result is:

```python
[
    {"chunk_size": 4, "prefill_steps": 3, "step_token_loads": [4, 4, 2]},
    {"chunk_size": 6, "prefill_steps": 2, "step_token_loads": [6, 4]},
    {"chunk_size": 10, "prefill_steps": 1, "step_token_loads": [10]}
]
```

## What the gate checks

The gate builds an oracle using the same chunked-prefill step-count model. It
computes the expected prefill steps with integer arithmetic and derives every
per-step token load from the remaining prompt length.

The returned list of dictionaries must exactly match the oracle output for
multiple prompt lengths and chunk-size sweeps. Any approximation, sorting, or
incorrect handling of the final remainder fails.
