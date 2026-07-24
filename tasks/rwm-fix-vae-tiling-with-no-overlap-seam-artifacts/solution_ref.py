import numpy as np


def tiled_decode(image, decode_fn, tile_size, overlap):
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape
    # reflect-pad the WHOLE image once, at its true boundary -- every
    # tile then reads real neighboring pixels (or the same reflected
    # boundary values the untiled decode would use), never fabricating
    # context from its own tile alone.
    padded_image = np.pad(image, overlap, mode="reflect")
    out = np.zeros((H, W), dtype=np.float64)

    for r0 in range(0, H, tile_size):
        for c0 in range(0, W, tile_size):
            r1 = min(r0 + tile_size, H)
            c1 = min(c0 + tile_size, W)
            # patch = core tile + `overlap` pixels of real context on
            # every side, taken from the globally-padded image.
            patch = padded_image[r0:r1 + 2 * overlap, c0:c1 + 2 * overlap]
            decoded = decode_fn(patch)
            # decode_fn's own (unknown) receptive-field shrink, inferred
            # from how much smaller it made the patch.
            shrink_r = (patch.shape[0] - decoded.shape[0]) // 2
            shrink_c = (patch.shape[1] - decoded.shape[1]) // 2
            # crop off the leftover context band -- what remains lines
            # up exactly with this tile's core region.
            cr = overlap - shrink_r
            cc = overlap - shrink_c
            core = decoded[cr:decoded.shape[0] - cr, cc:decoded.shape[1] - cc]
            out[r0:r1, c0:c1] = core

    return out
