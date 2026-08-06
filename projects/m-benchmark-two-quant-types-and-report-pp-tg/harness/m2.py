import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from llambench.parser import extract_quant_metrics, parse_llama_bench_json
    from llambench.roofline import compute_throughput_ratio, predict_tg_throughput

    out = {"roofline_match": 0.0, "throughput_ratio": 0.0}
    
    item = ref.TEST_BENCHMARKS[0]
    parsed = parse_llama_bench_json(item["raw_json"])
    metrics = extract_quant_metrics(parsed, "Q4_K_M")
    
    m_bytes = item["model_bytes"]["Q4_K_M"]
    bw = item["bandwidth_gbps"]
    
    want_pred = ref.reference_predict_tg(m_bytes, bw)
    got_pred = predict_tg_throughput(m_bytes, bw)
    
    want_ratio = ref.reference_ratio(metrics["tg"], want_pred)
    got_ratio = compute_throughput_ratio(metrics["tg"], got_pred)
    
    if abs(want_pred - got_pred) < 1e-2:
        out["roofline_match"] = 1.0
    else:
        out["_note"] = f"Predicted TG mismatch: got {got_pred}, expected {want_pred}"

    out["throughput_ratio"] = float(got_ratio)
    return out
