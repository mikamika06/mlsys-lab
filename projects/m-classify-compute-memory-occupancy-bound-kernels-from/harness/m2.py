import ref

def check(workdir):
    from trace_parser.torch_trace import analyze_torch
    out = {"torch_match": 0.0}

    got = analyze_torch(ref.TORCH_FIXTURE, ref.TORCH_FLOPS)
    want = ref.analyze_torch(ref.TORCH_FIXTURE, ref.TORCH_FLOPS)

    if got and want and set(got.keys()) == set(want.keys()):
        diffs = [abs(got[k] - want[k]) for k in want]
        if max(diffs) < 1e-4:
            out["torch_match"] = 1.0

    return out
