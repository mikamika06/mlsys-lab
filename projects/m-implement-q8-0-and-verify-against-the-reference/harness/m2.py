import ref

def check(workdir):
    from quant.legacy import block_properties, rank_legacy_types
    try:
        props = block_properties()
        if not isinstance(props, dict) or "Q8_0" not in props:
            return {"properties_match": 0.0, "_note": "missing Q8_0 in properties"}
        if props["Q8_0"]["bytes_per_block"] != 34:
            return {"properties_match": 0.0, "_note": "incorrect bytes_per_block for Q8_0"}

        weights = ref.generate_test_weights()
        ranking = rank_legacy_types(weights)
        if not isinstance(ranking, list) or len(ranking) == 0:
            return {"properties_match": 0.0, "_note": "invalid ranking output"}
        return {"properties_match": 1.0}
    except Exception as e:
        return {"properties_match": 0.0, "_note": str(e)}
