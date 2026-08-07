## Context

CUDA graph capture reduces launch overhead by recording a sequence of GPU operations and replaying that sequence with fixed memory addresses. A common pattern is to allocate static input and output buffers during capture.

A replay does not create new buffers. Instead, each call copies new inputs into the static input buffer, executes the captured computation, and writes the result into the static output buffer. If the caller needs to keep a result after another replay, the result must be copied out before the next replay overwrites the static output storage.

This task models a captured computation:

$$Y = XW^T$$

where $X$ is an input matrix and $W$ is a fixed weight matrix.

The important semantic property is that the internal output buffer is reused across replays, but every returned result must be an independent snapshot.

## Task

Implement `static_buffer_replay(W)`.

```python
def static_buffer_replay(W: list[list[float]]):
    ...
```

The function receives a list of lists of floats `W` with shape $(m, d)$ and returns a callable:

```python
replay = static_buffer_replay(W)
```

The returned `replay(X)` accepts an input array `X` with shape $(n, d)$ and returns the matrix:

$$Y = XW^T$$

The implementation must behave like a CUDA graph replay:

1. Allocate fixed internal input and output buffers when `static_buffer_replay` is called.
2. On every `replay(X)` call, copy `X` into the static input buffer.
3. Compute the output into the static output buffer.
4. Return a snapshot that is not modified by later calls.

The returned array may not alias the internal output buffer.

## Example

```python

W = [[1., 2.], [3., 4.]]
replay = static_buffer_replay(W)

a = replay([[1., 0.]])
b = replay([[0., 1.]])

# a remains unchanged after the second replay:
# a == [[1., 3.]]
# b == [[2., 4.]]
```

## What the gate checks

The gate builds a Python oracle that models the captured graph behavior. It runs a sequence of replays and stores snapshots from the oracle before the next replay overwrites the static output buffer.

The student's complete sequence of returned outputs is compared with the oracle using maximum absolute error:

$$\max_i |y_i - \hat{y_i}| \le 10^{-12}$$

Returning the internal output array directly fails because previous outputs change after later replays.
