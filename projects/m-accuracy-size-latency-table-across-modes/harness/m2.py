import ref


def check(workdir):
    from quanteval.table import build_ptq_summary_table, rank_ptq_modes

    out = {"table_correct": 0.0, "size_ratio": 1.0, "ranking_correct": 0.0}
    layers = ref.make_sample_layers(seed=456)
    hw = ref.make_hw_config()
    inputs = ref.make_test_inputs(seed=456)

    try:
        table = build_ptq_summary_table(layers, hw, inputs)
    except Exception as e:
        out["_note"] = f"build_ptq_summary_table raised: {e}"
        return out

    ref_table = ref.build_ptq_summary_table_ref(layers, hw, inputs)

    table_ok = True
    for mode in ["fp32", "fp16", "dynamic_int8", "full_int8"]:
        if mode not in table:
            table_ok = False
            out["_note"] = f"mode {mode} missing from summary table"
            break
        ref_m = ref_table[mode]
        got_m = table[mode]
        if got_m["size_bytes"] != ref_m["size_bytes"]:
            table_ok = False
            out["_note"] = f"{mode} size_bytes mismatch: got {got_m['size_bytes']}, want {ref_m['size_bytes']}"
            break
        if abs(got_m["size_ratio"] - ref_m["size_ratio"]) > 1e-4:
            table_ok = False
            out["_note"] = f"{mode} size_ratio mismatch: got {got_m['size_ratio']}, want {ref_m['size_ratio']}"
            break
        if abs(got_m["latency_us"] - ref_m["latency_us"]) > 1e-3:
            table_ok = False
            out["_note"] = f"{mode} latency mismatch: got {got_m['latency_us']}, want {ref_m['latency_us']}"
            break
        if abs(got_m["mse"] - ref_m["mse"]) > 1e-3:
            table_ok = False
            out["_note"] = f"{mode} mse mismatch: got {got_m['mse']}, want {ref_m['mse']}"
            break

    if table_ok:
        out["table_correct"] = 1.0

    out["size_ratio"] = float(table.get("full_int8", {}).get("size_ratio", 1.0))

    try:
        ranked = rank_ptq_modes(table, max_mse_threshold=1.0)
        ref_ranked = ref.rank_ptq_modes_ref(table, max_mse_threshold=1.0)
        if ranked == ref_ranked:
            out["ranking_correct"] = 1.0
        else:
            out["_note"] = f"ranking mismatch: got {ranked}, expected {ref_ranked}"
    except Exception as e:
        out["_note"] = f"rank_ptq_modes raised: {e}"

    return out
