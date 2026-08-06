def identify_schedule(actions):
    types = [a["type"] for a in actions]
    if any("WEIGHT" in t for t in types):
        return "zero_bubble"
    if any("VIRTUAL" in t for t in types):
        return "interleaved"
    fw_indices = [i for i, t in enumerate(types) if t == "FORWARD"]
    bw_indices = [i for i, t in enumerate(types) if t == "BACKWARD"]
    if fw_indices and bw_indices and max(fw_indices) < min(bw_indices):
        return "gpipe"
    return "1f1b"
