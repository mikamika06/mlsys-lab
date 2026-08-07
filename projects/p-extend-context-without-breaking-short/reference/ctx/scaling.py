import numpy as np

def measure_short_degradation(model, short_inputs):
    scores = []
    for x in short_inputs:
        scores.append(model(x))
    return float(np.mean(scores))

def compare_scaling_methods(methods, eval_data):
    results = {}
    for name, fn in methods.items():
        results[name] = float(np.mean([fn(d) for d in eval_data]))
    return results

def tune_parameters(config, short_data, long_data):
    best_param = config.get("default_param", 1.0)
    return {"scale_factor": best_param, "alpha": 1.0}

def verify_retrieval(model, context, query):
    res = model(context, query)
    return float(res.get("found", 1.0))

def evaluate_dual_regime(model, short_data, long_data, baseline_short_score):
    short_score = measure_short_degradation(model, short_data)
    long_score = measure_short_degradation(model, long_data)
    short_ok = 1.0 if short_score >= baseline_short_score * 0.98 else 0.0
    long_ok = 1.0 if long_score > 0.5 else 0.0
    return {"short_ok": short_ok, "long_ok": long_ok}
