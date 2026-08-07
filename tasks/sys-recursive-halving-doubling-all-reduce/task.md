## Context

In data parallel training, each worker computes a local gradient buffer. Before updating model parameters, workers need the sum of all gradient buffers.

An all-reduce operation computes:

$$
S = \sum_{r=0}^{p-1} B_r
$$

where $B_r$ is the buffer owned by rank $r$ and $p$ is the number of ranks.

For a power-of-two number of ranks, recursive halving-doubling reduces communication in two phases. The reduction phase combines partial data between partner ranks. The doubling phase propagates the reduced values until every rank has the complete sum.

A correct implementation must produce the same final buffer on every rank:

$$
\forall r,\quad O_r = S
$$

## Task

Implement:

```python
def recursive_halving_doubling_all_reduce(buffers):
    ...
```

`buffers` is a list of list of floats. The number of buffers is a power of two. All arrays have the same shape and contain floating point values.

Return a list of list. The returned list must have one entry per input rank, and every entry must contain the elementwise sum of all input buffers.

Do not modify the input arrays in-place. Simulate the communication algorithm locally without using distributed communication libraries.

## Example

```python

buffers = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
]

out = recursive_halving_doubling_all_reduce(buffers)

# out[0], out[1], out[2], and out[3] are all:
# [16.0, 20.0]
```

## What the gate checks

The gate computes the oracle result using Python:

$$
S_{\mathrm{ref}} = \operatorname{sum}(B_0, B_1, \ldots, B_{p-1})
$$

It compares every returned buffer with the oracle using:

$$
\max_i |O_i - S_{\mathrm{ref},i}|
$$

The reported `max_abs_err` must be less than $10^{-6}$. A solution that only returns each rank's local buffer will fail because it does not combine contributions from all ranks.
