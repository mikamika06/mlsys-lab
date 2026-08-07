def check(workdir):
    import ref
    from lora_pipe import engine
    m = {"quant_ok": 0.0}
    weights = ref.get_base_weights()
    try:
        q_out = engine.quantize_model(weights, bits=4)
        if isinstance(q_out, dict) and "weights" in q_out and "scales" in q_out:
            m["quant_ok"] = 1.0
    except Exception:
        pass
    return m
