from llambench.parser import extract_quant_metrics, parse_llama_bench_json
from llambench.roofline import compute_throughput_ratio, predict_tg_throughput


def test_bench_metrics_integrity():
    json_str = '[{"n_prompt": 512, "n_gen": 0, "avg_ts": 1200.0, "quant": "Q4_K_M"}, {"n_prompt": 0, "n_gen": 128, "avg_ts": 45.0, "quant": "Q4_K_M"}]'
    parsed = parse_llama_bench_json(json_str)
    metrics = extract_quant_metrics(parsed, "Q4_K_M")
    assert metrics["pp"] > metrics["tg"], "Prompt processing throughput must be higher than text generation throughput"
    
    predicted = predict_tg_throughput(4.0 * 1e9, 200.0)
    ratio = compute_throughput_ratio(metrics["tg"], predicted)
    assert ratio > 0.0, "Throughput ratio must be positive"
