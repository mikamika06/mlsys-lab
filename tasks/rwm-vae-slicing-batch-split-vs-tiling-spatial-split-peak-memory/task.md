## Context

VAE decoders used in image generation systems can reduce peak memory by splitting the
decode workload. Two common strategies are batch slicing and spatial tiling.

For a batch of $B$ images with decoded activation size $H \times W \times C$, batch
slicing decodes one image at a time. The peak decoded activation memory is therefore
proportional to

$$M_{\mathrm{slice}} = HWC.$$

Spatial tiling keeps the batch dimension but processes only one spatial tile at a
time. For tile dimensions $h_t \times w_t$, the peak decoded activation memory is

$$M_{\mathrm{tile}} = B h_t w_t C.$$

The lower value indicates the strategy with the smaller peak activation footprint.
This simplified model matches the key memory tradeoff used by production image
generation pipelines: batch splitting reduces the number of samples in flight, while
tiling reduces the spatial area in flight.

## Task

Implement `compare_vae_decode_memory`:

```python
def compare_vae_decode_memory(
    batch: int,
    height: int,
    width: int,
    channels: int,
    tile_height: int,
    tile_width: int,
) -> tuple[int, int, str]:
    ...
```

Return a tuple containing:

1. The peak activation element count for batch slicing.
2. The peak activation element count for spatial tiling.
3. The strategy name with the smaller peak: either `"slicing"` or `"tiling"`.

All values are counts of activation elements, not bytes. Assume all activations have
the same dtype and that dimensions are positive integers.

## Example

```python
result = compare_vae_decode_memory(4, 1024, 1024, 4, 256, 256)

# (
#   4194304,
#   1048576,
#   "tiling"
# )
```

## What the gate checks

The gate builds several decode configurations and computes the expected peaks with an
independent oracle implementation of the memory formulas. The returned tuple must
exactly match the oracle output for every case.

A solution that confuses batch slicing with spatial tiling, ignores the batch
dimension, or chooses the strategy before comparing the two computed peaks will fail.
