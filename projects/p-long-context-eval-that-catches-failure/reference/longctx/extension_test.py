import numpy as np

def compare_extension_methods(method_a_results, method_b_results):
    score_a = np.mean([r["success"] for r in method_a_results])
    score_b = np.mean([r["success"] for r in method_b_results])
    return {"method_a_score": float(score_a), "method_b_score": float(score_b), "comparison_ok": True}
