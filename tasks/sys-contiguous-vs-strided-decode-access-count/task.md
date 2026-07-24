## Context

A transformer decoder reads keys and values from a KV cache during generation. The
logical cache shape can be represented as $[b,h,s,d]$, where $b$ is batch size,
$h$ is the number of attention heads, $s$ is the cached sequence length, and
$d$ is the head dimension.

The same logical tensor can use different physical memory layouts. A contiguous
layout stores dimensions in C order as $[b,h,s,d]$. A strided layout can store
the same logical view by transposing a base array with shape $[b,s,h,d]$.

During a decode step, one attention head reads its full cached sequence. For this
task the modeled read is:

$$
(0, 0, 0 \ldots s-1, 0 \ldots d-1).
$$

For every element address $a$, the cache line index is

$$
\left\lfloor \frac{a}{64} \right\rfloor .
$$

The memory-access count is the number of unique cache lines touched by this
decode read. Different layouts can produce different counts because the same
logical sequence can be separated by larger physical strides.

## Task

Implement `decode_access_count(shape, layout)`:

```python
def decode_access_count(shape: tuple[int, int, int, int], layout: str) -> int:
    ...
```

The input `shape` is $(b,h,s,d)$. Return the number of 64-byte cache lines
touched when reading the first batch and first head across all sequence
positions and head-dimension values.

Support these layouts:

- `"contiguous"`: create a NumPy C-contiguous array with shape $(b,h,s,d)$.
- `"strided"`: create a NumPy array with the same logical shape by transposing a
  base array with shape $(b,s,h,d)$.

Use the actual NumPy strides of the selected layout. The result must be an
integer.

## Example

```python
decode_access_count((1, 8, 64, 8), "contiguous")
# returns the cache-line count for reading one head in a contiguous KV cache

decode_access_count((1, 8, 64, 8), "strided")
# returns a larger count because sequence positions are separated by head stride
```

## What the gate checks

The gate builds real NumPy arrays for each layout and computes the reference
cache-line count from their actual strides. Your implementation is compared
against this oracle on multiple shapes.

A solution that always assumes C-contiguous memory will fail because the
transposed layout changes the physical addresses accessed by the decode read.
