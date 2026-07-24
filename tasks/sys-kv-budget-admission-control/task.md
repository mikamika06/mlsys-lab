## Context

Continuous batching systems run many generation requests together. Each active request
uses key-value (KV) cache memory. A scheduler must decide which waiting requests can
enter a decoding iteration while keeping the KV budget within a fixed limit.

Assume each request $r_i$ has a KV cost $k_i$ and a remaining token count $t_i$.
A batch has budget $B$. During one scheduling step, requests are admitted in queue
order until adding the next request would make the total KV usage exceed the budget.

The admitted set is a prefix of the waiting queue. If the currently admitted KV usage is
$S$, a request with cost $k_i$ is accepted only when

$$S + k_i \leq B.$$

Completed requests are removed after each step. A request completes when its remaining
token count reaches zero after one decoding iteration.

## Task

Implement `admit_requests(requests, budget)`:

```python
def admit_requests(requests: list[dict], budget: int) -> list[list[int]]:
    ...
```

The input is a list of request dictionaries. Each dictionary has:

- `"id"`: integer request identifier.
- `"kv"`: integer KV cache cost.
- `"tokens"`: integer number of decoding iterations remaining.

The function simulates iteration-level scheduling. At every step:

1. Consider requests that have not completed.
2. Scan them in their original input order.
3. Admit requests while the running KV total stays within `budget`.
4. Decode one token for every admitted request.
5. Remove requests whose token count reaches zero.
6. Record the admitted request IDs for this step.

Return a list of steps. Each inner list contains the IDs admitted during that step, in admission order. Do not mutate the input request dictionaries.

## Example

```python
requests = [
    {"id": 1, "kv": 4, "tokens": 2},
    {"id": 2, "kv": 3, "tokens": 1},
    {"id": 3, "kv": 5, "tokens": 1},
]

print(admit_requests(requests, 7))
# [[1, 2], [1], [3]]
```

The first step admits requests 1 and 2 because $4 + 3 \leq 7$. Request 2 finishes
after that step. Request 1 continues alone. Request 3 waits because admitting it with
request 1 would exceed the budget.

## What the gate checks

The gate compares the returned schedule with a reference scheduler generated from the
same admission rules. It checks the complete step-by-step admission order, including
which requests remain active and when they finish.
