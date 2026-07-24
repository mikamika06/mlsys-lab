## Context

Distributed training overlaps backward computation with gradient communication. A gradient bucket can start an all-reduce only after every gradient inside that bucket has been produced by the backward pass.

Assume gradients arrive in a fixed order. Gradient $i$ becomes ready at time

$$
r_i = \sum_{j=0}^{i} c_j ,
$$

where $c_j$ is the compute time for gradient $j$.

A bucket containing gradients from index $s$ through $t$ has ready time

$$
R(s,t)=\max_{i=s}^{t} r_i
$$

and communication duration

$$
T(s,t)=\lambda+\frac{\sum_{i=s}^{t} b_i}{B},
$$

where $b_i$ is the gradient size, $B$ is communication bandwidth, and $\lambda$ is fixed communication latency per bucket.

Buckets communicate sequentially on a single communication stream. A good partition starts communication early enough to hide it behind remaining backward computation, while avoiding too many latency costs.

## Task

Implement `bucket_grads(grad_sizes, compute_times, bandwidth, latency, max_bucket_bytes)`.

Arguments:

- `grad_sizes`: a list of gradient sizes in bytes.
- `compute_times`: a list of backward compute times for the same gradients.
- `bandwidth`: communication bandwidth in bytes per unit time.
- `latency`: fixed communication latency for each bucket.
- `max_bucket_bytes`: maximum allowed total bytes in one bucket.

Return a list of lists containing gradient indices. Every gradient must appear exactly once. Each bucket must be contiguous in the original gradient order and must not exceed `max_bucket_bytes`.

The objective is to minimize exposed communication time: the amount of communication that remains after the final backward computation has completed.

## Example

```python
buckets = bucket_grads(
    [40, 30, 50],
    [2.0, 2.0, 2.0],
    bandwidth=100.0,
    latency=0.1,
    max_bucket_bytes=80,
)

# One valid output shape:
# [[0], [1], [2]]
```

The exact answer depends on the scheduling objective, not on a fixed bucket count.

## What the gate checks

The gate computes the optimal partition with an exhaustive dynamic-programming oracle on small cases. It simulates backward readiness and serialized communication using the equations above.

The returned partition is scored by

$$
\mathrm{size\_ratio} =
\frac{\mathrm{exposed\_comm}_{candidate}}
{\mathrm{exposed\_comm}_{optimal}} .
$$

The ratio must satisfy $\mathrm{size\_ratio}\leq 1.05$. Solutions that use only a greedy bucket fill strategy can fail because they may create buckets that start too late or pay unnecessary latency.
