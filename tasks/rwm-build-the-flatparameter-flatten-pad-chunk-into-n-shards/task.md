## Context

FSDP (Fully Sharded Data Parallel) doesn't shard each tensor individually
— it builds one big **FlatParameter**: every parameter tensor in a module
is flattened to 1-D and concatenated into a single buffer, so the whole
module's memory becomes one contiguous block. That block is then padded
with zeros so its length is an exact multiple of `world_size` (the number
of ranks), and split into `world_size` equal-size contiguous shards, one
per rank. Padding is what makes "equal-size" possible for an arbitrary
total parameter count — without it, ranks would get uneven shard sizes
and the collective communication (all-gather / reduce-scatter) that FSDP
relies on would break, since those ops require equal-size buffers per
rank.

For a flattened buffer of length $T$ and `world_size` $N$:

$$
\text{pad} = (N - (T \bmod N)) \bmod N, \qquad
\text{shard\_size} = \frac{T + \text{pad}}{N}
$$

## Task

Implement `flatten_pad_shard`:

```python
def flatten_pad_shard(params, world_size):
    ...
```

- `params`: a list of arrays of arbitrary shape (a module's parameters,
  in order).
- `world_size`: number of shards $N$, a positive `int`.

Steps:
1. Ravel every array in `params`, in order, and concatenate into one 1-D
   `float64` buffer of length $T$.
2. Zero-pad the buffer at the **end** to the next multiple of
   `world_size` (0 extra elements if $T$ is already a multiple of $N$).
3. Split the padded buffer into `world_size` equal-length contiguous
   shards, each of length `shard_size` as above.

Return a list of `world_size` 1-D `float64` arrays, all the same length.

## Example

```python

params = [[1.0, 2.0, 3.0], [[4.0, 5.0]]]
# flatten+concat: [1, 2, 3, 4, 5]  (T = 5)
flatten_pad_shard(params, world_size=2)
# pad = (2 - 5%2) % 2 = 1 -> padded = [1, 2, 3, 4, 5, 0]  (length 6)
# shard_size = 6 / 2 = 3
# -> [array([1., 2., 3.]), array([4., 5., 0.])]
```

## What the gate checks

The grader builds several seeded cases — module parameter lists of mixed
shapes against different `world_size` values, including one case where
the total element count already divides evenly (zero padding expected)
and one single-parameter, `world_size=1` case — and computes the oracle
flatten/pad/shard independently in Python for each, never calling your
function.

`size_ok` requires your output to be exactly `world_size` arrays, each of
the oracle's `shard_size` length, across every case (must be `>= 1.0`,
i.e. every case correct) — this catches wrong padding math (off-by-one
shard count or size) before even comparing values. `byte_exact_fraction`
concatenates your shards and the oracle's shards and requires the byte
buffers to match exactly (must be `>= 1.0`) — since flatten/pad/chunk is
pure indexing and zero-fill with no rounding involved, any correct
implementation reproduces the oracle bit-for-bit; a mismatch means wrong
element order, wrong padding position/value, or misaligned shard
boundaries.
