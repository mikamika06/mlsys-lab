def check(workdir):
    m = {"qdq_relocated": 0.0}
    try:
        from int8_eng.tuning import relocate_qdq
        graph = {"qdq_nodes": ["layer1", "sensitive_layer"]}
        res = graph.copy()
        relocate_qdq(res, ["sensitive_layer"])
        if "sensitive_layer" not in res.get("qdq_nodes", []):
            m["qdq_relocated"] = 1.0
    except Exception:
        pass
    return m
