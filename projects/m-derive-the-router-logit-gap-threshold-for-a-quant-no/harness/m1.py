import ref
from routerquant.analysis import compute_quant_error


def check(workdir):
    weights, quantized_weights, _, _ = ref.generate_inputs()
    want = compute_quant_error(weights, quantized_weights)
    from routerquant import analysis
    try:
        got = analysis.compute_quant_error(weights, quantized_weights)
    except Exception as e:
        return {"error_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    if not isinstance(got, dict) or "mse" not in got or "max_error" not in got:
        return {"error_matched": 0.0, "_note": "missing keys"}
    diff_mse = abs(got["mse"] - want["mse"])
    diff_max = abs(got["max_error"] - want["max_error"])
    if diff_mse < 1e-5 and diff_max < 1e-5:
        return {"error_matched": 1.0}
    return {"error_matched": 0.0, "_note": f"got {got}, want {want}"}
