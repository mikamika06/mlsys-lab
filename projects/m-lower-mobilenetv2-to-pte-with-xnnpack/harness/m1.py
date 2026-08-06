import ref

def check(workdir):
    from edge_export.lower import lower_model
    out = {"lower_match": 0.0}
    for cfg in ref.CONFIGS:
        want = {"format": "pte", "nodes": [
            {"name": "conv1", "target": "xnnpack", "op": "conv2d"},
            {"name": "bn1", "target": "cpu_fallback", "op": "batch_norm"},
            {"name": "relu1", "target": "xnnpack", "op": "relu6"},
            {"name": "block1_dw", "target": "xnnpack", "op": "conv2d"},
            {"name": "block1_pw", "target": "xnnpack", "op": "conv2d"},
            {"name": "add1", "target": "xnnpack", "op": "add"},
            {"name": "custom_op", "target": "cpu_fallback", "op": "unsupported_custom"}
        ]}
        got = lower_model(cfg)
        if got == want:
            out["lower_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    return out
