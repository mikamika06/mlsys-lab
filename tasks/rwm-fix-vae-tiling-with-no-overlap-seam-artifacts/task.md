## Context

A VAE decoder (or any convolutional network) can't hold a huge latent in
memory as one pass, so production image pipelines decode it in tiles and
stitch the results back together. Every conv layer only has **local
support** — it needs some radius $r$ of neighboring pixels to compute
each output pixel correctly. If you cut the image into non-overlapping
tiles and decode each one *in isolation*, the network has no real
neighbors to look at near a tile's own edges — it has to fabricate
context (typically by padding the tile with its own reflected border),
which does **not** match what the true neighboring pixels (just across
the tile boundary, in the next tile) would have produced. The result is
a visible discontinuity — a seam — at every internal tile border, even
though each tile's *interior* decoded perfectly fine.

The fix is to give every tile `overlap` extra pixels of **real** context
on each side before decoding — sourced from the actual neighboring
image content (or, only at the image's true edge, the same reflect
padding the untiled decode itself would use there) — then crop the
decoded patch back down to the tile's core region before placing it in
the output. As long as `overlap` is at least as large as the decoder's
receptive-field radius $r$, this reconstructs the untiled decode
exactly.

## Task

Fix `tiled_decode`:

```python
def tiled_decode(image, decode_fn, tile_size, overlap):
    ...
```

- `image`: `(H, W)` `float64` array.
- `decode_fn`: a callable with **local support only** — given any 2-D
  array, it returns an array shrunk by some fixed (to you, unknown)
  margin $s$ on every side, e.g. a `(h, w)` input returns
  `(h - 2s, w - 2s)`. It never sees anything outside the array it's
  handed.
- `tile_size`: side length of each tile's **core** region in the
  output (the stride at which tiles are placed along each axis; the
  last row/column of tiles may be smaller if `tile_size` doesn't evenly
  divide `H`/`W`).
- `overlap`: extra pixels of real image context to include on every
  side of a tile before calling `decode_fn` ($\text{overlap} \geq s$ is
  guaranteed by the caller).

The supplied version ignores `overlap` and locally reflect-pads each
tile using only that tile's own pixels — producing seams. Fix it so
each core tile at rows $[r_0, r_1)$, columns $[c_0, c_1)$ is decoded
using a patch that includes `overlap` pixels of genuine context on
every side (reflected only at the true image boundary, exactly like the
untiled decode would), then cropped back to the core region before
being written into the output. Since `decode_fn`'s shrink margin $s$ is
not told to you directly, infer it from the shapes:
$s = (\text{patch.shape}[i] - \text{decoded.shape}[i]) / 2$.

Return the reconstructed `(H, W)` `float64` array.

## Example

```python

def blur5(patch):  # local support, shrinks by 2 on every side
w = [[[[patch[i + r][j + c] for c in range(5)] for r in range(5)] for j in range(len(patch[0]) - 5 + 1)] for i in range(len(patch) - 5 + 1)]
    return w.mean(axis=(-1, -2))

_rng = **import**('random').Random(0); image = [[_rng.gauss(0, 1) for _ in range(24)] for _ in range(24)]
out = tiled_decode(image, blur5, tile_size=8, overlap=4)
out.shape  # (24, 24) -- matches the untiled decode of `image`, no seams
```

## What the gate checks

The grader builds several seeded `(H, W)` images with different
`(tile_size, overlap)` pairs, including a case where `H`/`W` aren't
multiples of `tile_size` (uneven last tile) and one where `overlap`
equals the decoder's minimum receptive-field radius exactly. For each
it computes the untiled reference by reflect-padding the *whole* image
once at its true boundary and decoding it in one shot — never calling
your function.

`max_abs_err` is the worst-case max elementwise absolute difference
between your stitched output and the untiled reference, across all
cases (must be `<= 1e-6`). Because the decoder has purely local support
and `overlap` always covers its receptive field, a correct
overlap-aware tiling reconstructs the untiled decode to floating-point
precision — while the no-overlap (locally-padded) version differs by
tens of percent at every internal tile border, since it feeds the
decoder fabricated edge pixels instead of the real neighboring image
content.
