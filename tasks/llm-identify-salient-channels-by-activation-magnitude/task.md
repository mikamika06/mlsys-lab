## Context

In neural networks, each neuron in a layer produces an *activation* value for every input sample.
For a given batch of activations $X \in \mathbb{R}^{n_{\text{samples}}\times n_{\text{channels}}}$,
the magnitude of a channel can be summarised by the mean absolute activation
$$m_j = \frac{1}{n_{\text{samples}}}\sum_{i=1}^{n_{\text{samples}}} \vert{}X_{ij}\vert{}\,.$$
Channels with large $m_j$ are often considered *salient* because they contribute more strongly to downstream computations.

## Task

Implement the function `salient_channels`:

```python
def salient_channels(X: list[list[float]], fraction: float = 0.1) -> list[int]:
    ...
```

The function must:

1. Accept a 2‑D list `X` of shape $(n_{\text{samples}}, n_{\text{channels}})$.
2. Compute the mean absolute activation per channel as described above.
3. Return a 1‑D integer list containing the indices of the top $\lceil \text{fraction}\times n_{\text{channels}}\rceil$ channels with the largest $m_j$.
4. The returned indices must be sorted in ascending order.
5. If `fraction` is zero, return an empty list.
6. Raise a `ValueError` if `fraction` is not in the interval $[0,1]$.


The output elements should be int.

## Example

```python
X = [[ 1, -2,  3],
              [ 4, -5,  6]]
# mean abs per channel: [2.5, 3.5, 4.5]
# top 50% -> ceil(0.5*3)=2 channels: indices 1 and 2
idx = salient_channels(X, fraction=0.5)
print(idx)   # [1, 2]
```

## What the gate checks

The grader computes a reference set of indices using Python operations on the same input. It then compares your output to this reference with `==`.

If they match exactly, the `exact_match` metric is 1.0; otherwise it is 0.0.
