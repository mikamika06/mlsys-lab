import importlib

import ref


def check(workdir):
    from benchkit import parse

    mod = importlib.import_module("tests.test_bench")
    out = {"flags_real_anomaly": 0.0, "clean_data_silent": 0.0}
    if not hasattr(mod, "audit"):
        return out

    rows = parse.load_all(ref.files())
    warnings = mod.audit(rows)
    if isinstance(warnings, list) and any("depth" in str(w).lower() for w in warnings):
        out["flags_real_anomaly"] = 1.0

    clean = []
    for depth, ts in ((0, 10.0), (512, 9.0), (2048, 8.0)):
        clean.append({"model_type": "toy", "n_prompt": 0, "n_gen": 64,
                      "n_depth": depth, "n_ubatch": 512, "n_batch": 512,
                      "avg_ns": int(64 / ts * 1e9), "stddev_ns": 0,
                      "avg_ts": ts, "stddev_ts": 0.0,
                      "samples_ns": [int(64 / ts * 1e9)] * 3,
                      "samples_ts": [ts, ts, ts], "model_size": 1, "model_n_params": 1})
    for i, r in enumerate(clean):
        r["_source"] = "clean.json"
        r["_row"] = i
    if mod.audit(clean) == []:
        out["clean_data_silent"] = 1.0
    return out
