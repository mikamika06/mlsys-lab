import ref


def check(workdir):
    from inductorsched.trace import parse_fusion_trace
    import inductorsched.trace as ref_trace

    samples = ref.get_trace_samples()
    ok = 0
    for sample in samples:
        want = ref_trace.parse_fusion_trace(sample)
        got = parse_fusion_trace(sample)
        if got == want:
            ok += 1

    matched = 1.0 if ok == len(samples) else 0.0
    return {"traces_matched": matched}
