def compare_vae_decode_memory(
    batch,
    height,
    width,
    channels,
    tile_height,
    tile_width,
):
    slicing_peak = height * width * channels
    tiling_peak = batch * tile_height * tile_width * channels
    strategy = "slicing" if slicing_peak <= tiling_peak else "tiling"
    return slicing_peak, tiling_peak, strategy
