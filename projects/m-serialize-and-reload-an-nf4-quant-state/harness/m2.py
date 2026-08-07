import ref

def check(workdir):
    from quantutil.config import rebuild_quantization_config

    out = {"quantization_prevented": 0.0, "weights_preserved": 0.0}
    try:
        meta = ref.METADATA_SAMPLES[0]
        cfg = rebuild_quantization_config(meta)
        if isinstance(cfg, dict) and cfg.get("bits") == 4:
            out["quantization_prevented"] = 1.0
            out["weights_preserved"] = 1.0
    except Exception:
        pass
    return out
