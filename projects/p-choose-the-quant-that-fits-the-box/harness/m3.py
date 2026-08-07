def check(workdir):
    import sys
    import numpy as np
    sys.path.insert(0, workdir)
    from quant.analyzer import measure_quality
    import ref

    m = {"quality_metrics_ok": 0.0}
    try:
        logits_ref = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        logits_quant = np.array([[1.1, 1.9, 3.0], [0.6, 1.4, 2.5]])
        res = measure_quality(logits_ref, logits_quant)
        ores = ref.oracle_measure_quality(logits_ref, logits_quant)
        if abs(res["kld"] - ores["kld"]) < 1e-5 and abs(res["ppl"] - ores["ppl"]) < 1e-5:
            m["quality_metrics_ok"] = 1.0
    except Exception:
        pass
    return m
