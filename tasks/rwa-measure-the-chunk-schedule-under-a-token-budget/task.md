## Context

In many language‑model pipelines a prompt of length $L$ tokens must be processed in chunks that respect a maximum number of batched tokens per step, denoted $\mathsf{budget}$. The scheduler splits the prompt into consecutive steps so that each step processes at most $\mathsf{budget}$ tokens. The last step may contain fewer tokens if $L$ is not an exact multiple of $\mathsf{budget}$.

The number of required steps is therefore
$$
\mathsf{steps} = \left\lceil \frac{L}{\mathsf{budget}} \right\rceil,
$$
and the per‑step prefill token counts are all equal to $\mathsf{budget}$ except for the final step, which receives the remaining tokens:
$$
c_i =
\begin{cases}
\mathsf{budget} & 1 \le i < \mathsf{steps},\\[4pt]
L - \mathsf{budget}\,( \mathsf{steps}-1) & i = \mathsf{steps}.
\end{cases}
$$

## Task

Implement the function `chunk_schedule` that, given a prompt length `prompt_len` and an integer token budget `max_tokens_per_step`, returns a tuple `(num_steps, prefill_counts)` where:

* `num_steps` is the number of steps required to process the entire prompt.
* `prefill_counts` is a list of length `num_steps` containing the exact number of tokens processed in each step.

The function must handle any non‑negative integer inputs. If `prompt_len` is zero, return `(0, [])`.

## Example

```python
>>> chunk_schedule(10, 4)
(3, [4, 4, 2])

>>> chunk_schedule(8, 4)
(2, [4, 4])
```

## What the gate checks

The grader computes a reference schedule using the exact formulas above and compares it to the student's output with an `exact_match` metric. The comparison is strict: any deviation in step count or token counts causes failure.
