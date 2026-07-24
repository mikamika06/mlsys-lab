## Context

Continuous batching systems serve multiple generation requests at the same time. A scheduler may stop a request temporarily when memory pressure requires its state to be evicted, then restore it later.

A generation request can be modeled as a sequence of tokens produced by repeatedly applying a deterministic transition function. If the full uninterrupted execution produces tokens

$$
t_0, t_1, \dots, t_{k-1},
$$

then a correct preemption and resume implementation must produce the same sequence even when execution is split into multiple runs. The scheduler may change the order in which requests are advanced, but it must not change the per-request state.

For a request state $s_i$ and token transition function $f$:

$$
s_{i+1} = f(s_i, t_i).
$$

A preemption event saves the current state. Resuming must continue from that saved state rather than restarting from the initial state or sharing another request's state.

## Task

Implement `resume_decode(requests, quantum)`:

```python
def resume_decode(requests, quantum):
    ...
```

`requests` is a list of dictionaries. Each dictionary contains:

- `"id"`: an integer request identifier.
- `"seed"`: an integer initial state value.
- `"steps"`: the number of tokens to generate.

The function must return a dictionary mapping each request id to its generated token list.

The scheduler must simulate iteration-level batching:

1. Maintain independent state for every request.
2. Run at most `quantum` token-generation iterations before rotating to the next active request.
3. Requests with completed generation are removed from the active set.
4. A resumed request must continue from its saved state.

The token transition is:

$$
x_{n+1} = (1103515245x_n + 12345) \bmod 2^{31},
$$

and the emitted token is:

$$
t_n = x_{n+1} \bmod 1000.
$$

Do not restart a request after preemption. Different requests must not share mutable generation state.

## Example

```python
requests = [
    {"id": 7, "seed": 1, "steps": 4},
    {"id": 8, "seed": 2, "steps": 3},
]

out = resume_decode(requests, 1)

# Equivalent to uninterrupted decoding:
# {
#   7: [590, 575, 84, 781],
#   8: [635, 610, 303]
# }
```

## What the gate checks

The gate computes an uninterrupted decoding oracle from the transition equation and compares the returned token streams with the oracle.

The `exact_match` metric must equal $1.0$. Any lost state, duplicated token, restarted request, or cross-request state sharing changes the final stream and fails the gate.
