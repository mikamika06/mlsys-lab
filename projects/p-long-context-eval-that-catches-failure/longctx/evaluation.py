import numpy as np
from longctx.dataset import generate_dataset
from longctx.extrapolation import evaluate_extrapolation
from longctx.metrics import compute_positional_curve

def run_evaluation(context_len=1000, model_type="faulty"):
    tasks = generate_dataset(context_len, num_samples=10)
    if model_type == "faulty":
        res = evaluate_extrapolation(tasks, method="linear")
    else:
        res = evaluate_extrapolation(tasks, method="ntk")
    curve = compute_positional_curve(res)
    tok_fail = 0.0
    attn_fail = 1.0 if curve["dip_detected"] else 0.0
    return {"curve": curve, "tokenization_failure": tok_fail, "attention_failure": attn_fail, "caught": curve["dip_detected"]}
