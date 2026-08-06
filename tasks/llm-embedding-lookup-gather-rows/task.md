## Context

In language models the embedding matrix $E \in \mathbb{R}^{V\times D}$ stores a dense vector for every token in the vocabulary.  
For an input sequence of token ids $\mathbf{t} = (t_1,\dots,t_n)$ the model needs to retrieve the corresponding rows:
$$
\mathbf{y}_i = E_{t_i}\quad i=1,\dots,n .
$$
This operation is often called *gather* or *lookup*.  It must be performed efficiently and with exact numerical precision.

## Task

Implement a function that performs this lookup:

```python
def lookup_embeddings(ids: list[int], weights: list[list[float]]) -> list[list[float]]:
    ...
```

- `ids` – a list of floats of integer token ids.  
- `weights` – the embedding matrix of shape `(V, D)` with dtype `float64`.  
The function must return an array of shape `(len(ids), D)` containing the rows indexed by `ids`, also as `float64`.

## Example

```python
E = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
ids = [2, 0]
out = lookup_embeddings(ids, E)
# out == [[0.5, 0.6],
#         [0.1, 0.2]]
```

## What the gate checks

The grader compares your output with a reference implementation that uses Python’s advanced indexing.  
It reports the maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_{i,j}\lvert \hat y_{ij} - y_{ij}\rvert .
$$

Your solution must achieve $\mathrm{max\_abs\_err}\le 10^{-7}$.
