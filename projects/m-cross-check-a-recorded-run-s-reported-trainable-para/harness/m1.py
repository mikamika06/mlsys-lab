import ref


def check(workdir):
    from loraparams.formula import calculate_trainable_params

    out = {"configs_matched": 0.0}
    ok = 0
    for i in range(len(ref.MODEL_CONFIGS)):
        m_cfg = ref.MODEL_CONFIGS[i]
        l_cfg = ref.LORA_CONFIGS[i]
        want = ref.ref_calculate_trainable_params(m_cfg, l_cfg)
        try:
            got = calculate_trainable_params(m_cfg, l_cfg)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"cfg {i} raised {type(e).__name__}: {e}"
    out["configs_matched"] = float(ok)
    return out
