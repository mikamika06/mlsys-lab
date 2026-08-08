import ref


def check(workdir):
    from kernelstats.metrics import compute_arithmetic_intensity

    out = {"intensities_matched": 0.0}
    oracle = ref.get_oracle_data()
    match_count = 0
    for i, item in enumerate(ref.TRACES):
        want = ref.get_oracle_data()[i]["intensity"]
        got = compute_arithmetic_intensity(item["flops"], item["bytes_transferred"])
        if abs(got - want) < 1e-5:
            match_count += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got {got}, want {want}"

    if match_count == len(ref.TRACES):
        out["intensities_matched"] = 1.0
    return out
