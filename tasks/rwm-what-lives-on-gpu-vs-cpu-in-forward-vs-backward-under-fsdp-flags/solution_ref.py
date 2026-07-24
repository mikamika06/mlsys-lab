def residency(cpu_offload: bool,
              activation_checkpoint: bool,
              activation_offload: bool) -> dict:
    """
    Return a mapping from phase names to lists of items that are on the GPU.

    Parameters
    ----------
    cpu_offload : bool
        If True, parameters (`shard` and `full_param`) are offloaded to CPU.
    activation_checkpoint : bool
        If True, activations are not stored after forward; they are recomputed
        during backward and thus only appear in the backward list.
    activation_offload : bool
        If True (and checkpointing is False), activations are offloaded to CPU.

    Returns
    -------
    dict
        Keys are `"forward"` and `"backward"`.  Values are sorted lists of
        items that reside on GPU during that phase.
    """
    forward = []
    backward = []

    # Parameters residency
    if not cpu_offload:
        forward.extend(["shard", "full_param"])
        backward.extend(["shard", "full_param"])

    # Activations residency
    if activation_checkpoint:
        # Recomputed on GPU during backward
        backward.append("activations")
    else:
        if not activation_offload:
            forward.append("activations")
            backward.append("activations")

    return {"forward": sorted(forward), "backward": sorted(backward)}
