import numpy as np


def _ramp(n):
    if n <= 0:
        return np.array([])
    return (np.arange(n) + 1) / (n + 1)


def tiled_vae_decode(z: np.ndarray, decode_fn, tile_size: int, overlap: int) -> np.ndarray:
    """
    Decode a latent in overlapping tiles (to keep the decoder's peak
    activation bounded) and combine them with a linear-ramp blend over the
    overlap band so no tile seam is visible.
    """
    z = np.asarray(z, dtype=np.float64)
    H, W, Cin = z.shape

    # Probe the output channel count with a tiny call.
    probe_h = min(1 + 2 * overlap, H)
    probe_w = min(1 + 2 * overlap, W)
    probe = decode_fn(z[:probe_h, :probe_w, :])
    Cout = probe.shape[-1]

    out = np.zeros((H, W, Cout), dtype=np.float64)
    weight = np.zeros((H, W, 1), dtype=np.float64)

    for i0 in range(0, H, tile_size):
        for j0 in range(0, W, tile_size):
            i1 = min(i0 + tile_size, H)
            j1 = min(j0 + tile_size, W)

            ei0 = max(0, i0 - overlap)
            ei1 = min(H, i1 + overlap)
            ej0 = max(0, j0 - overlap)
            ej1 = min(W, j1 + overlap)

            patch = z[ei0:ei1, ej0:ej1, :]
            decoded = decode_fn(patch)
            core = decoded[i0 - ei0:i1 - ei0, j0 - ej0:j1 - ej0, :]

            th, tw = i1 - i0, j1 - j0
            ramp_h = min(overlap, tile_size)

            wy = np.ones(th)
            if i0 > 0:
                r = _ramp(ramp_h)[:th]
                wy[:len(r)] = np.minimum(wy[:len(r)], r)
            if i1 < H:
                r = _ramp(ramp_h)[:th]
                wy[-len(r):] = np.minimum(wy[-len(r):], r[::-1])

            wx = np.ones(tw)
            if j0 > 0:
                r = _ramp(ramp_h)[:tw]
                wx[:len(r)] = np.minimum(wx[:len(r)], r)
            if j1 < W:
                r = _ramp(ramp_h)[:tw]
                wx[-len(r):] = np.minimum(wx[-len(r):], r[::-1])

            wtile = (wy[:, None] * wx[None, :])[:, :, None]

            out[i0:i1, j0:j1, :] += core * wtile
            weight[i0:i1, j0:j1, :] += wtile

    return out / np.maximum(weight, 1e-12)
