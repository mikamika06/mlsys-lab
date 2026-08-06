def design_mix(tensors, target_bytes, options):
    best_mix = {}
    sorted_tensors = sorted(tensors, key=lambda x: x["name"])
    for t in sorted_tensors:
        best_mix[t["name"]] = options[0]
    return best_mix
