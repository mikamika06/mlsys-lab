## Context

Constrained decoding is used in production generation systems to prevent invalid
tokens from being sampled. A finite state machine (FSM) tracks the current decode
state. Each state has a set of allowed vocabulary tokens, and all other tokens
are masked by setting their logits to $-\infty$.

For a vocabulary of size $V$, if a state allows $k$ tokens, the mask density is

$$
\rho = \frac{V-k}{V}.
$$

A decode trace is a sequence of FSM states visited during generation. Measuring
the mask density at every step shows how restrictive the constraint is. The mean
density over a trace is

$$
\bar{\rho} = \frac{1}{T}\sum_{t=1}^{T}\rho_t,
$$

where $T$ is the number of decode steps.

## Task

Implement `mask_density_trace`:

```python
def mask_density_trace(vocab_size, trace, allowed):
    ...
```

Arguments:

- `vocab_size` is the integer vocabulary size $V$.
- `trace` is a list of FSM state ids visited during decoding.
- `allowed` is a dictionary mapping each state id to a collection of allowed token ids.

Return a tuple:

```python
(densities, mean_density)
```

where:

- `densities` is a NumPy array of shape $(T,)$ containing the mask density for
  every state in `trace`.
- `mean_density` is the arithmetic mean of `densities` as a Python float.

The implementation should compute the fraction of vocabulary tokens masked out,
not the fraction of tokens allowed.

## Example

```python
trace = [0, 1, 0]
allowed = {
    0: [2, 5],
    1: [1, 2, 3, 4]
}

densities, mean_density = mask_density_trace(10, trace, allowed)

# densities = [0.8, 0.6, 0.8]
# mean_density = 0.7333333333333333
```

## What the gate checks

The gate builds FSM decode traces and allowed-token tables, then recomputes the
mask densities directly from the FSM definition. The returned values are
compared with the oracle using relative error $\mathrm{rel\_err}$.

The gate requires

$$
\mathrm{rel\_err} < 10^{-9}.
$$
