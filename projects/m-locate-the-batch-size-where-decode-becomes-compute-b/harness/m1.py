import sys

def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from roofline.analysis import find_decode_compute_bound_batch_size

    out = {"transitions_matched": 0.0}
    ok = 0
    for cfg in ref.MODEL_CONFIGS:
        for hw in ref.HARDWARE_SPECS:
            want = ref.find_decode_compute_bound_batch_size(cfg, hw)
            got = find_decode_compute_bound_batch_size(cfg, hw)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {cfg['name']} hw {hw['name']}: got {got}, reference {want}"

    out["transitions_matched"] = float(ok)
    return out
