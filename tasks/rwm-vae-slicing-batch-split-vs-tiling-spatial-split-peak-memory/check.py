def _oracle(batch, height, width, channels, tile_height, tile_width):
    full_image = batch * 0 + height * width * channels
    tiled = batch * tile_height * tile_width * channels
    if full_image <= tiled:
        choice = "slicing"
    else:
        choice = "tiling"
    return (int(full_image), int(tiled), choice)


def grade(sol, fx) -> dict:
    cases = [
        (4, 1024, 1024, 4, 256, 256),
        (8, 512, 768, 8, 128, 192),
        (2, 2048, 2048, 4, 512, 512),
        (16, 256, 256, 4, 64, 64),
        (1, 768, 768, 8, 384, 384),
        (6, 640, 960, 4, 160, 240),
    ]
    for args in cases:
        try:
            got = sol.compare_vae_decode_memory(*args)
        except Exception:
            return {"exact_match": 0.0}
        if tuple(got) != _oracle(*args):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
