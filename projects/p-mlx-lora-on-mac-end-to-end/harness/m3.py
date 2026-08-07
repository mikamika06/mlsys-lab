def check(workdir):
    import ref
    from lora_pipe import engine
    import numpy as np
    m = {"merge_ok": 0.0}
    base = ref.get_base_weights()
    adapter = {"layer1": {"A": np.ones((8, 8))*0.1, "B": np.ones((8, 8))*0.1}}
    try:
        merged = engine.merge_adapter(base, adapter)
        if "layer1" in merged and merged["layer1"].shape == (8, 8):
            m["merge_ok"] = 1.0
    except Exception:
        pass
    return m
