from servemetrics.bench import compute_tpot_overhead

def generate_report(config, seed=42):
    res = compute_tpot_overhead(config, seed=seed)
    return f"Schema: {config['schema_complexity']} | Unconstrained TPOT: {res['mean_unconstrained_tpot']:.2f}ms | Guided TPOT: {res['mean_guided_tpot']:.2f}ms | Ratio: {res['latency_ratio']:.2f}x"
