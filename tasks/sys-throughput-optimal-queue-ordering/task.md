## Context

Continuous batching systems serve multiple generation requests by repeatedly choosing which
waiting requests enter an execution batch. A scheduler must balance two constraints:

- The total KV-cache memory of active requests must stay within a fixed budget.
- The order of admitting requests affects the number of modeled decoding steps needed to
  complete the queue.

For this task, each request $r_i$ has:

- a prompt length $p_i$,
- a generation length $g_i$,
- a KV cost per active token $k_i$.

The request consumes

$$
m_i = k_i(p_i + g_i)
$$

units of KV memory when fully resident. A batch can contain requests whose total memory does
not exceed the budget $B$.

The modeled execution cost of an ordering is the number of iteration steps required by a
simple continuous batching simulator. At each step, all active requests generate one token.
A request leaves the active set after producing $g_i$ tokens. New requests are admitted in
the supplied queue order whenever their memory fits.

The goal is to find an ordering that minimizes the total number of simulated steps:

$$
C = \sum_{t=1}^{T} 1 ,
$$

where $T$ is the number of iterations until all requests complete.

## Task

Implement `schedule_queue(requests, kv_budget)`.

The input `requests` is a list of tuples:

```python
(prompt_tokens, generation_tokens, kv_per_token)
```

The function must return a list of integer indices describing the order in which waiting
requests should be admitted.

The returned list must contain every index exactly once. The scheduler may choose any order,
but the resulting simulated step count should be close to the minimum possible ordering.

You may assume that test queues are small enough that an optimal scheduler can be found by
enumerating candidate orderings.

## Example

```python
requests = [
    (10, 4, 2),
    (5, 8, 1),
    (3, 2, 1),
]

order = schedule_queue(requests, 40)
# one valid high-throughput ordering is returned as a list of indices
```

## What the gate checks

The gate builds several queues and computes the true optimum by exhaustively evaluating every
possible ordering with the same simulator definition. The returned ordering is simulated and
compared with that optimum.

The reported metric is

$$
\mathrm{size\_ratio} =
\frac{\text{candidate step count}}
{\text{optimal step count}} .
$$

The solution passes when $\mathrm{size\_ratio} \le 1.05$.
