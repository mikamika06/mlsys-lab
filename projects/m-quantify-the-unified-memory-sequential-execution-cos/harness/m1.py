import ref


def check(workdir):
    from seqcost.model import model_bytes
    from seqcost.memory import kv_bytes

    max_err = 0.0
    for cfg in ref.CONFIGS:
        w_model = ref.model_bytes(cfg)
        g_model = model_bytes(cfg)
        if w_model > 0:
            err_m = abs(w_model - g_model) / w_model
            max_err = max(max_err, err_m)

        w_kv = ref.kv_bytes(cfg, 1024)
        g_kv = kv_bytes(cfg, 1024)
        if w_kv > 0:
            err_k = abs(w_kv - g_kv) / w_kv
            max_err = max(max_err, err_k)

    return {"rel_err": float(max_err)}
