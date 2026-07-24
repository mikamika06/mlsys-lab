def compute_peak_vram(num_layers: int, layer_size: int, batch_size: int) -> dict:
    """
    Compute the peak VRAM usage for three diffusion‑model memory strategies.

    Parameters
    ----------
    num_layers : int
        Number of layers in the model.
    layer_size : int
        Size (in bytes) of a single token for one layer.
    batch_size : int
        Number of tokens processed simultaneously.

    Returns
    -------
    dict
        Keys:
            full_resident          – peak MB when all weights and activations are resident.
            model_offload          – peak MB when the whole model is off‑loaded to CPU.
            sequential_offload     – peak MB when layers are streamed one at a time.
    """
    W = num_layers * layer_size
    A = layer_size * batch_size
    M = 1024**2  # bytes per megabyte

    return {
        "full_resident": (W + A) / M,
        "model_offload": max(W, A) / M,
        "sequential_offload": max(layer_size, A) / M
    }
