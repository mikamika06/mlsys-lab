## Context

During LLM serving, prefill work and decode work compete for execution resources. A chunked prefill scheduler limits the amount of prompt processing performed in one chunk so that decode requests do not experience excessive stalls.

A simplified model uses a chunk size $c$ tokens. The number of prefill chunks is

$$
k = \left\lceil \frac{P}{c} \right\rceil ,
$$

where $P$ is the total number of prefill tokens. The time spent on one prefill chunk is modeled as

$$
t_p(c) = \alpha c^2 + \beta c ,
$$

and the decode work delayed by each chunk is modeled as

$$
t_d(c) = \gamma c .
$$

The total modeled decode stall is therefore

$$
S(c) = k \cdot t_d(c) + \max(0, k \cdot t_p(c) - B),
$$

where $B$ is the available token budget window. The first term represents decode delay caused by each prefill chunk, while the second term penalizes chunk schedules that exceed the allowed budget.

The goal is not to guess a fixed chunk size. The scheduler should evaluate valid chunk sizes and choose the one with the lowest modeled stall.

## Task

Implement `select_chunk_size(total_tokens, budget, alpha, beta, gamma, max_chunk)`:

```python
def select_chunk_size(
    total_tokens: int,
    budget: float,
    alpha: float,
    beta: float,
    gamma: float,
    max_chunk: int,
) -> int:
    ...
```

Return the integer chunk size $c$ that minimizes the modeled stall $S(c)$.

Valid chunk sizes are all integers satisfying $1 \le c \le \min(total_tokens, max_chunk)$.

If multiple chunk sizes have the same minimum stall, return the smallest chunk size.

## Example

```python
chunk = select_chunk_size(
    total_tokens=100,
    budget=5000.0,
    alpha=0.01,
    beta=0.5,
    gamma=2.0,
    max_chunk=64,
)

# chunk is the best integer chunk size under the model
```

## What the gate checks

The gate computes the true optimum by evaluating every valid chunk size with the same mathematical model. It then compares the stall produced by the submitted chunk size against that optimum.

The reported metric is

$$
\mathrm{size\_ratio} = \frac{S(c_{\mathrm{submitted}})}{S(c_{\mathrm{optimal}}) + 10^{-12}} .
$$

A passing solution must have $\mathrm{size\_ratio} \le 1.05$ on all hidden cases. A solution that always returns a common chunk size will fail because the optimum changes with the workload parameters.
