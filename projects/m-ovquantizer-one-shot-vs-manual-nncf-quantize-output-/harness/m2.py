import ref


def check(workdir):
    from quantizer.pipeline import verify_output_parity

    matched = 0.0
    max_rel_err = 0.0

    for case in ref.CASES:
        config = {
            "nodes": case["nodes"],
            "graph_opset_map": case["graph_opset_map"],
            "num_bits": 8,
            "symmetric": True,
            "calibration_data": case["calib"],
        }
        res = verify_output_parity(case["weights"], case["inputs"], config)
        rel_err = float(res.get("rel_err", 1.0))
        max_rel_err = max(max_rel_err, rel_err)

        if res.get("is_exact_match") and rel_err <= 1e-4:
            matched += 1.0

    return {
        "parity_matched": matched,
        "max_rel_err": max_rel_err,
    }
