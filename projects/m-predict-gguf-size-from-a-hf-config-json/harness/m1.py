import ref


def check(workdir):
    from ggufsize.config import parse_config
    out = {"tensors_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.get_tensor_counts(cfg)
        try:
            got = parse_config(cfg)
        except Exception:
            got = {}

        if set(want.keys()) == set(got.keys()):
            match = True
            for k, shape in want.items():
                if tuple(got.get(k, ())) != tuple(shape):
                    match = False
                    break
            if match:
                ok += 1
    out["tensors_matched"] = float(ok)
    return out
