def residency(mode: str):
    """
    Return a mapping from each pipeline stage to the set of components that are
    resident on the GPU for the given offload mode.

    Parameters
    ----------
    mode : str
        One of "model", "sequential" or "group".

    Returns
    -------
    dict[str, set[str]]
        Keys are "encode", "denoise", "decode".  Each value is a set containing
        any subset of {"text_encoder", "denoiser", "vae"} that should be on the GPU.
    """
    stages = ["encode", "denoise", "decode"]
    if mode == "model":
        comp_set = {"text_encoder", "denoiser", "vae"}
        return {s: set(comp_set) for s in stages}
    elif mode == "sequential":
        mapping = {
            "encode": {"text_encoder"},
            "denoise": {"denoiser"},
            "decode": {"vae"},
        }
        return mapping
    elif mode == "group":
        mapping = {
            "encode": {"text_encoder", "denoiser"},
            "denoise": {"denoiser", "vae"},
            "decode": {"vae"},
        }
        return mapping
    else:
        raise ValueError(f"Unknown mode: {mode}")
