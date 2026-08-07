import ref

def check(workdir):
    m = {"bits8_ok": 0.0}
    try:
        from qlora_fix.optimizer import quantize_optimizer_to_8bit
        states = {"param1": [1, 2, 3]}
        res = quantize_optimizer_to_8bit(states)
        if "param1" in res and res["param1"].get("bits") == 8:
            m["bits8_ok"] = 1.0
    except Exception:
        pass
    return m
