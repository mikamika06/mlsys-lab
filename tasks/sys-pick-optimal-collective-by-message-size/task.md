## Context

Collective communication algorithms trade off startup latency and data transfer cost. A tree-based collective usually has fewer communication steps, while a ring-based collective can achieve better bandwidth utilization for large messages.

For a message of size $m$ bytes, consider the simplified costs:

$$
T_{\mathrm{tree}}(m) = \alpha \lceil \log_2(p) \rceil + \beta m \frac{p-1}{p}
$$

and

$$
T_{\mathrm{ring}}(m) = \alpha (p-1) + \beta m \frac{2(p-1)}{p},
$$

where $p$ is the number of participating processes, $\alpha$ is the per-step latency cost, and $\beta$ is the per-byte transfer cost.

The optimal collective is the one with the smaller estimated cost. The decision boundary depends on both the message size and the communication parameters.

## Task

Implement `pick_collective`:

```python
def pick_collective(message_sizes, processes, alpha, beta):
    ...
```

The function receives:

- `message_sizes`: an iterable of non-negative message sizes in bytes.
- `processes`: the number of participating processes $p$.
- `alpha`: the latency coefficient.
- `beta`: the bandwidth coefficient.

Return a list of strings with one entry per message size. Each entry must be either `"tree"` or `"ring"` and must select the algorithm with the lower cost according to the model above. If the costs are equal, return `"tree"`.

Do not use external libraries.

## Example

```python
result = pick_collective(
    [64, 1024, 1048576],
    processes=8,
    alpha=10.0,
    beta=0.001,
)

# result is:
# ["tree", "ring", "ring"]
```

## What the gate checks

The gate builds independent cases and computes the expected choices by evaluating the communication cost model directly in the grader. It compares the returned list against this computed reference using exact matching.

A solution that uses an incorrect threshold, reverses the comparison, or ignores the process count will fail.
