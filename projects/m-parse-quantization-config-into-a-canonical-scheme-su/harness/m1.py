import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from quantlib.scheme import parse_quantization_config

    out = {"configs_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.parse_quantization_config(cfg)
        got = parse_quantization_config(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"Expected {want}, got {got}"

    if ok == len(ref.CONFIGS):
        out["configs_matched"] = 1.0
    return out
