import ref

def check(workdir):
    m = {"accuracy_loss_ok": 0.0}
    try:
        from cpuopt.quantizer import calibrate_and_quantize
        data = ref.get_sample_data()
        res = calibrate_and_quantize(None, data)
        if isinstance(res, dict) and res.get("quantized") and res.get("accuracy_loss", 1.0) < 0.02:
            m["accuracy_loss_ok"] = 1.0
    except Exception:
        pass
    return m
