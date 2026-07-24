import numpy as np


def tiled_decode(image: np.ndarray, decode_fn, tile_size: int, overlap: int) -> np.ndarray:
    """Tile-decode `image` and stitch the tiles back into a full image
    that matches the untiled decode of the whole image.

    image: (H, W) float64 array.
    decode_fn: a callable with LOCAL support only -- given any 2-D
        array, it returns a same-family array shrunk by a fixed (but
        callee-unknown) margin on every side, e.g. `decode_fn(patch)`
        might return a `(patch.shape[0]-4, patch.shape[1]-4)` array.
        It never sees anything outside the array it's handed.
    tile_size: side length of each tile's CORE region in the output
        (the stride at which tiles are placed; the last row/col of
        tiles may be smaller if tile_size doesn't divide H or W).
    overlap: extra pixels of real image context to include on every
        side of a tile before calling decode_fn, so that decode_fn's
        receptive field at the tile's edges sees genuine neighboring
        pixels instead of the tile's own (fabricated) boundary.

    Returns the reconstructed (H, W) float64 array. BUG: this
    implementation ignores `overlap` and pads each tile with its own
    reflected edge instead of sourcing real neighboring pixels from the
    rest of the image -- producing visible discontinuities at every
    internal tile border.
    """
    H, W = image.shape
    out = np.zeros((H, W), dtype=np.float64)
    for r0 in range(0, H, tile_size):
        for c0 in range(0, W, tile_size):
            r1 = min(r0 + tile_size, H)
            c1 = min(c0 + tile_size, W)
            tile = image[r0:r1, c0:c1].astype(np.float64)
            padded = np.pad(tile, 2, mode="reflect")
            decoded = decode_fn(padded)
            out[r0:r1, c0:c1] = decoded
    return out
