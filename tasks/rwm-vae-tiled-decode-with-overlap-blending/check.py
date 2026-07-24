import numpy as np


def _conv3x3(tile, W, b):
    H, Wd, Cin = tile.shape
    Cout = W.shape[-1]
    padded = np.pad(tile, ((1, 1), (1, 1), (0, 0)))
    out = np.zeros((H, Wd, Cout), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            out += padded[di:di + H, dj:dj + Wd, :] @ W[di, dj]
    return out + b


def _make_decoder(seed, cin, cmid, cout):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((3, 3, cin, cmid)) * 0.3
    b1 = rng.standard_normal((cmid,)) * 0.05
    W2 = rng.standard_normal((3, 3, cmid, cout)) * 0.3
    b2 = rng.standard_normal((cout,)) * 0.05

    def raw_decode(tile):
        h = np.tanh(_conv3x3(tile, W1, b1))
        return np.tanh(_conv3x3(h, W2, b2))

    return raw_decode


def _make_capped_decode_fn(raw_decode, max_tile):
    def decode_fn(tile):
        h, w = tile.shape[0], tile.shape[1]
        if h > max_tile or w > max_tile:
            raise ValueError(
                f"tile too large ({h}x{w} > {max_tile}) -- would exceed peak activation budget"
            )
        return raw_decode(tile)

    return decode_fn


def _oracle_reference(raw_decode, z):
    return raw_decode(z)


def grade(sol, fx) -> dict:
    """
    For several (image size, tile_size, overlap, channel) configurations,
    builds a small deterministic local conv decoder, decodes the full latent
    once (unrestricted) as the reference, and compares it against the
    submission's tiled_vae_decode using a decode_fn that raises if asked to
    process anything bigger than tile_size + 2*overlap (so decoding the
    whole latent at once is not an option -- real tiling is required).
    """
    configs = [
        # (H, W, tile_size, overlap, Cin)
        (48, 48, 16, 8, 4),
        (40, 40, 16, 6, 3),
        (56, 48, 20, 8, 4),
        (37, 45, 16, 8, 3),
        (48, 48, 20, 6, 4),
        (33, 33, 12, 6, 3),
    ]
    cmid, cout = 6, 3

    errs = []
    for t, (H, W, tile_size, overlap, cin) in enumerate(configs):
        rng = np.random.default_rng(100 + t)
        z = rng.standard_normal((H, W, cin)).astype(np.float64)

        raw_decode = _make_decoder(seed=t, cin=cin, cmid=cmid, cout=cout)
        ref = _oracle_reference(raw_decode, z)

        max_tile = tile_size + 2 * overlap
        decode_fn = _make_capped_decode_fn(raw_decode, max_tile)

        try:
            got = sol.tiled_vae_decode(z.copy(), decode_fn, int(tile_size), int(overlap))
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            errs.append(float("inf"))
            continue

        if got.shape != ref.shape:
            errs.append(float("inf"))
            continue

        errs.append(float(np.max(np.abs(got - ref))))

    return {"max_abs_err": float(np.max(errs))}
