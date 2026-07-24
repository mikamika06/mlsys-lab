def _ref(cpu_offload, activation_checkpoint, activation_offload):
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

def grade(sol, fx) -> dict:
    cases = [
        (False, False, False),
        (True,  False, False),
        (False, True,  False),
        (False, False, True),
        (True,  True,  False),
        (True,  False, True),
        (False, True,  True),
        (True,  True,  True)
    ]
    for cpu_offload, activation_checkpoint, activation_offload in cases:
        try:
            got = sol.residency(cpu_offload, activation_checkpoint, activation_offload)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(cpu_offload, activation_checkpoint, activation_offload)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
