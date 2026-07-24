## Context

The **alpha-beta model** estimates the wall-clock cost of a point-to-point
message of $M$ bytes as

$$
T(M) = \alpha + \beta M
$$

where $\alpha$ is the fixed per-message latency (seconds) and $\beta$ is
the inverse bandwidth (seconds per byte). A collective operation across
$P$ ranks costs $(\text{steps}) \cdot \alpha + (\text{bytes moved per rank}) \cdot \beta$,
where "steps" and "bytes moved per rank" depend on the algorithm used:

- **`reduce_scatter`**, **`allgather`**, **`alltoall`** (ring / pairwise-exchange
  algorithm): $P-1$ sequential steps; each rank moves a $\frac{P-1}{P}$
  fraction of the total message $M$:
  $$
  T = (P-1)\,\alpha + \frac{P-1}{P}\,M\,\beta
  $$
- **`allreduce`** (ring, i.e. a `reduce_scatter` immediately followed by an
  `allgather`): twice the steps and twice the bandwidth term of the above:
  $$
  T = 2(P-1)\,\alpha + 2\,\frac{P-1}{P}\,M\,\beta
  $$
- **`broadcast`** (binomial tree): $\lceil \log_2 P \rceil$ steps, each
  moving the *full* message $M$ (no splitting):
  $$
  T = \lceil \log_2 P \rceil\,\alpha + \lceil \log_2 P \rceil\,M\,\beta
  $$

For `broadcast` with $P = 1$, no communication is needed, so $\text{steps} = 0$.

## Task

Implement `collective_cost`:

```python
def collective_cost(collective: str, num_ranks: int, message_bytes: float,
                     alpha: float, beta: float) -> dict:
    ...
```

- `collective`: one of `"allreduce"`, `"allgather"`, `"reduce_scatter"`,
  `"broadcast"`, `"alltoall"`.
- `num_ranks`: number of participating ranks $P \ge 1$.
- `message_bytes`: total logical payload size $M$ for the collective (the
  fully-gathered size for `allgather`/`allreduce`, the pre-reduction size
  for `reduce_scatter`, the full message for `broadcast`/`alltoall`).
- `alpha`: per-message latency in seconds.
- `beta`: per-byte transfer time in seconds/byte.

Use the formulas above for the named collective. Return a `dict`:

```python
{"latency_term": ..., "bandwidth_term": ..., "total": ...}
```

where `total == latency_term + bandwidth_term`.

## Example

```python
collective_cost("allreduce", num_ranks=4, message_bytes=1_000_000.0,
                 alpha=5e-6, beta=4e-10)
# steps = 2*(4-1) = 6            -> latency_term = 6 * 5e-6      = 3e-5
# frac = (4-1)/4 = 0.75          -> bandwidth_term = 2*0.75*1e6*4e-10 = 6e-4
# total = 3e-5 + 6e-4 = 0.00063
```

## What the gate checks

The grader evaluates every collective against a sweep of `(num_ranks,
message_bytes, alpha, beta)` configurations (including `num_ranks == 1`)
and computes the reference `latency_term`, `bandwidth_term`, and `total`
directly from the formulas above.

`modeled_mem_access` is the worst-case relative error, across all three
returned fields and all configurations, between your output and the
reference (must be `< 1e-9`). Using the wrong step count (e.g. `P` instead
of `P-1`), forgetting the factor of 2 for `allreduce`, or moving the full
message instead of the `(P-1)/P` fraction for the ring collectives will
all show up as a large relative error here.
