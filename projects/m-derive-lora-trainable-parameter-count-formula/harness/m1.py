import ref


def check(workdir):
    from lora.params import lora_params_count

    out = {"params_matched": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want = ref.param_count(cfg["in_features"], cfg["out_features"], cfg["rank"])
        got = lora_params_count(cfg["in_features"], cfg["out_features"], cfg["rank"])
        if got != want:
            ok = False
            out["_note"] = f"in {cfg['in_features']}, out {cfg['out_features']}, rank {cfg['rank']}: got {got}, want {want}"
            break
    if ok:
        out["params_matched"] = 1.0
    return out
