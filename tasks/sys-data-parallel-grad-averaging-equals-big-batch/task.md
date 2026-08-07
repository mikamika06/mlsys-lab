## Context

In data‑parallel training each worker processes a shard of the mini‑batch and computes the gradient on its local data. The gradients are then averaged across workers to obtain the update that would have been produced by a single “big” batch. For this simple linear regression problem with target vector $y$ and feature matrix $X$, the mean‑squared‑error loss is

$$
L(w) = \frac{1}{n}\sum_{i=1}^{n} (w^\top x_i - y_i)^2,
$$

whose gradient w.r.t. the weight vector $w$ is

$$
\nabla L(w)=\frac{2}{n}X^\top(Xw-y).
$$

If we initialise $w=\mathbf 0$, this reduces to

$$
g = \frac{2}{n}\,X^\top y .
$$

The data‑parallel gradient averaging rule states that if the batch is split into $S$ shards of equal size, each shard computes its local gradient

$$
g_s = \frac{2}{m} X_s^\top y_s,
$$

where $m=n/S$, and the global gradient is obtained by averaging:

$$
\bar g=\frac{1}{S}\sum_{s=1}^{S} g_s .
$$

Because all shards have the same number of samples, $\bar g$ equals the full‑batch gradient $g$.

## Task

Implement `data_parallel_grad_avg(X, y, num_shards)` that:

* accepts a 2‑D list `X` of shape $(n,d)$ and a 1‑D array `y` of length $n$,
* splits the data into `num_shards` equal shards (assume $n$ is divisible by `num_shards$),
* computes each shard’s gradient as described above, and
* returns the average of those gradients.

The function must use only vectorised Python operations; no explicit Python loops over samples. The returned gradient should be a 1‑D float64 array of length $d$.

## Example

```python
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0.5, 1.5, 2.5, 3.5]

grad = data_parallel_grad_avg(X, y, num_shards=2)
print(grad)   # → array([...])
```

## What the gate checks

The grader computes a reference gradient using Python and compares it to the user’s output with the scorer `max_abs_err`. The solution must satisfy

$$
\mathrm{max\_abs\_err} \le 10^{-6}.
$$

Any deviation larger than this threshold will fail the gate. No other metrics are evaluated.
