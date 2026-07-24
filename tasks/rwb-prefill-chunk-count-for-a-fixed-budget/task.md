## Context

Chunked prefill splits a long prompt's prefill work into pieces no larger
than a fixed token budget $C$, so each scheduling step's prefill cost is
bounded and can be mixed with ongoing decode work. A prompt of length $n$
tokens is processed in

$$
\text{num\_chunks} = \left\lceil \frac{n}{C} \right\rceil
$$

chunks: every chunk except possibly the last is exactly $C$ tokens, and the
last chunk holds whatever remains:

$$
\text{last\_chunk} = n - (\text{num\_chunks} - 1)\cdot C .
$$

($\text{last\_chunk} = C$ exactly when $n$ is a multiple of $C$ — there is
no dangling zero-sized chunk.)

## Task

Implement `chunk_counts`:

```python
def chunk_counts(prompt_lens: np.ndarray, chunk_budget: int) -> dict:
    ...
```

- `prompt_lens` — 1-D integer array of prompt lengths $n_i \ge 1$.
- `chunk_budget` — $C \ge 1$, the fixed per-chunk token budget.

Return a `dict` with:

- `"num_chunks"` — integer array, $\lceil n_i / C \rceil$ for each prompt.
- `"last_chunk"` — integer array, the last chunk's token count for each
  prompt, per the formula above.

## Example

```python
import numpy as np

chunk_counts(np.array([1, 512, 513, 1024]), chunk_budget=512)
# {"num_chunks": [1, 1, 2, 2], "last_chunk": [1, 512, 1, 512]}
```

## What the gate checks

The grader computes $\lceil n_i/C \rceil$ and the corresponding last-chunk
size directly with NumPy for several prompt-length fixtures and several
values of $C$, and requires **both** returned arrays to match the
reference exactly, element-for-element (`exact_match == 1.0`). An
off-by-one in the ceiling division, or computing the last chunk as
$n \bmod C$ (wrong whenever $n$ is an exact multiple of $C$, where it
should read $C$, not $0$), will fail on at least one prompt length.
