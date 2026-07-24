## Context

Batching requests in language model serving improves throughput but increases
latency. A scheduler must reject batch sizes that violate latency service-level
agreements (SLAs).

For a batch size $b$, the time to first token (TTFT) is modeled as

$$
\mathrm{TTFT}(b) = t_0 + p \cdot c_p \cdot b ,
$$

where $t_0$ is fixed overhead, $p$ is the number of prompt tokens, and $c_p$ is
the prefill cost per prompt token and batch item.

The inter-token latency (ITL) is modeled as

$$
\mathrm{ITL}(b) = d_0 + g \cdot c_d \cdot b ,
$$

where $d_0$ is fixed decode overhead, $g$ is the number of generated tokens, and
$c_d$ is the decode cost per generated token and batch item.

A batch size is feasible only when both constraints hold:

$$
\mathrm{TTFT}(b) \leq \mathrm{SLA}_{\mathrm{TTFT}}
$$

and

$$
\mathrm{ITL}(b) \leq \mathrm{SLA}_{\mathrm{ITL}} .
$$

## Task

Implement `feasible_batch_sizes`:

```python
def feasible_batch_sizes(
    batch_sizes,
    prompt_tokens,
    gen_tokens,
    sla_ttft_ms,
    sla_itl_ms,
):
    ...
```

Return a list of Python booleans. The output length must equal the length of
`batch_sizes`.

Use the following fixed cost model:

- $t_0 = 20.0$ ms
- $c_p = 0.05$ ms per prompt token per batch item
- $d_0 = 5.0$ ms
- $c_d = 0.01$ ms per generated token per batch item

Each returned value indicates whether that batch size satisfies both latency
limits.

## Example

```python
print(feasible_batch_sizes([1, 4, 16], 100, 50, 30, 15))
```

Output:

```python
[True, True, False]
```

## What the gate checks

The gate computes the expected result from the latency equations using a NumPy
oracle and compares the returned boolean mask exactly. Cases include situations
where TTFT passes but ITL fails, so ignoring either SLA constraint does not pass.
