def tiled_decode(image, decode_fn, tile_size, overlap):
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
    raise NotImplementedError('your code here')
