## Context

In many machine‑learning frameworks a *static shape capture* phase records the sizes of tensors that will appear during training or inference.  
When a batch size is chosen at runtime, it must be rounded up to one of these captured sizes so that the compiled graph can be reused.  

Let $C = (c_1,\dots,c_k)$ be the sorted list of captured bucket sizes and let $B=(b_1,\dots,b_m)$ be the requested batch sizes.  
For each $b_i$ we need to find

$$
\hat{c}_i \;=\;\min\{\,c_j \in C \mid c_j \ge b_i\,\},
$$

the smallest captured bucket that is at least as large as the request.  
If no such bucket exists (i.e., $b_i$ exceeds all captured sizes) we mark the batch as *eager* by returning $\hat{c}_i = -1$.  
The padding required for a non‑eager batch is simply

$$
p_i \;=\;\hat{c}_i - b_i .
$$

This operation must be performed efficiently because it can occur frequently during training.

## Task

Implement the function `round_to_bucket`:

```python
def round_to_bucket(captured_sizes: list[int], batch_sizes: list[int]) -> tuple[list[int], list[int]]:
    ...
```

The function receives a sorted list of captured bucket sizes and a list of requested batch sizes.  
It must return two lists:
* `chosen_buckets`: the selected bucket for each request (or `-1` if eager).
* `padded_rows`: the number of rows that will be padded to reach the chosen bucket.

The implementation should run in linear time with respect to the total number of batch sizes and use only standard Python constructs. No external libraries are required.

## Example

```python
captured = [32, 64, 128]
batches   = [10, 33, 65, 200]

chosen, padded = round_to_bucket(captured, batches)
# chosen == [32, 64, 128, -1]
# padded == [22, 31, 63, 0]
```

## What the gate checks

The grader computes a reference solution using the same algorithm described above and compares it to your output.  
A single metric `exact_match` is used: the returned tuples must be identical element‑wise to the oracle’s result.  
No additional performance or style metrics are enforced.
