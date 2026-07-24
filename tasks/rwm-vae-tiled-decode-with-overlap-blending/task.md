## Context

Decoding a large image from a diffusion-model latent through a VAE decoder can
need more activation memory than is available, because a convolutional
decoder's peak memory scales with the *whole* spatial extent it processes at
once. Libraries such as `diffusers` (`AutoencoderKL.tiled_decode`) fix this by
decoding the latent in **overlapping tiles** instead of all at once:

1. Walk the latent on a grid with stride `tile_size` (the tiles' non‑overlapping
   core positions).
2. For each grid position, extract a *larger* patch that adds `overlap`
   extra latent pixels of context on every side (clipped at the image
   border) — this halo gives the decoder's convolutions enough context to
   compute the *correct* value even for pixels near the core tile's edge.
3. Run the (memory-limited) decoder on that patch only.
4. Because adjacent tiles' halos overlap, more than one tile produces a
   value for pixels in the overlap band. Combine them with a **linear ramp
   ("feathered") blend**: a pixel a distance $d$ (in $\{1,\dots,\text{overlap}\}$)
   inside a tile's border, on the side facing a neighboring tile, gets
   weight $\frac{d}{\text{overlap}+1}$ from that tile (rising linearly from
   the border towards the tile's interior, where the weight saturates at 1).
   A pixel's final value is the weighted average of every tile that covers
   it, using the product of its horizontal and vertical ramp weights
   (weight 1 unless the pixel is within `overlap` of a *non-image-boundary*
   tile edge):

$$
\hat{I}(p) = \frac{\sum_{t} w_t(p)\, D_t(p)}{\sum_{t} w_t(p)}
$$

where $D_t(p)$ is tile $t$'s decoded value at pixel $p$ and $w_t(p)$ is its
blend weight at $p$ (0 outside the tile's core+halo, the ramp value inside
the overlap band, 1 in the tile's interior).

## Task

Implement:

```python
def tiled_vae_decode(z: np.ndarray, decode_fn, tile_size: int, overlap: int) -> np.ndarray:
    ...
```

* `z` — latent array of shape `(H, W, Cin)`.
* `decode_fn` — a callable: `decode_fn(tile: np.ndarray) -> np.ndarray` maps a
  spatial patch of shape `(h, w, Cin)` to a decoded patch of shape
  `(h, w, Cout)` (same spatial size, decoder is a 'same'-padded stack of
  convolutions). **`decode_fn` raises `ValueError` if `h` or `w` exceeds
  `tile_size + 2*overlap`** — it represents a decoder that only has enough
  memory for a tile-plus-halo-sized activation, so calling it on the full
  `z` at once is not an option. You must tile.
* `tile_size` — positive `int`, the core tile stride.
* `overlap` — positive `int`, halo pixels added on each side before calling
  `decode_fn`, and the width of the linear-ramp blend band between adjacent
  tiles.

Return the reconstructed `(H, W, Cout)` image. For every grid tile:

1. Extract `z[i0-overlap : i1+overlap, j0-overlap : j1+overlap, :]`, clipped
   to the array bounds.
2. Call `decode_fn` on that patch.
3. Slice out the sub-array of the decoded patch corresponding to the core
   tile region `[i0:i1, j0:j1]`.
4. Accumulate it into the output with the linear-ramp blend weights
   described above (accumulate `weight * value` and `weight` separately,
   then divide at the end).

## Example

```python
import numpy as np

def decode_fn(tile):
    # any local, 'same'-padded decoder -- e.g. a small conv stack
    ...

z = np.random.default_rng(0).standard_normal((48, 48, 4))
image = tiled_vae_decode(z, decode_fn, tile_size=16, overlap=8)
image.shape   # -> (48, 48, Cout)
```

If you decoded the whole `z` at once (were that allowed) you would get the
same `image` back to floating-point precision — the halo gives every pixel
full, correct context, and blending two tiles that both computed the exact
same correct value for an overlap pixel is a no-op.

## What the gate checks

The **max_abs_err** gate builds several `(z, decode_fn)` pairs — `decode_fn`
is a small deterministic 2-layer 'same'-padded convolutional network (fixed
random weights per trial) with a spatial receptive-field radius of 2, capped
so it refuses tiles larger than `tile_size + 2*overlap` — with varying image
sizes (including ones not evenly divisible by `tile_size`), tile sizes and
overlaps (`overlap` is always large enough to fully cover the decoder's
receptive field). It compares your `tiled_vae_decode` output against an
*unrestricted* reference decode of the whole `z` at once, and requires the
max absolute pixel error across all trials to be at or below the threshold —
tight enough that skipping the halo, skipping the blend, or misplacing tiles
all fail it, but loose enough for ordinary floating-point round-off.
