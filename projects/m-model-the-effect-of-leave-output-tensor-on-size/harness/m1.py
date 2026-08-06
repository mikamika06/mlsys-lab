import ref


def check(workdir):
    from ggufsize.model import parse_tensors, is_output_tensor

    out = {"shapes_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.parse_tensors(cfg)
        got = parse_tensors(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i} tensors mismatch"
    out["shapes_matched"] = float(ok)
    return out
