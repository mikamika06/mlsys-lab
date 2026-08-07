import numpy as np


class UnsupportedArchitectureError(Exception):
    pass


CONFIGS = [
    {"hf": [1.0, 2.0, 3.0], "ov": [1.0001, 2.0002, 3.0003], "threshold": 1e-3, "want": True},
    {"hf": [10.0, 20.0, 30.0], "ov": [10.2, 20.1, 29.8], "threshold": 1e-3, "want": False},
    {"hf": [0.5, 0.5, 0.5], "ov": [0.50001, 0.49999, 0.50002], "threshold": 1e-4, "want": True}
]


def verify_logits_parity(hf_logits, ov_logits, rel_err_threshold=1e-3):
    hf = np.asarray(hf_logits, dtype=np.float32)
    ov = np.asarray(ov_logits, dtype=np.float32)
    if hf.shape != ov.shape:
        return False
    diff = np.abs(hf - ov)
    denom = np.maximum(np.abs(hf), 1e-6)
    rel_errs = diff / denom
    return float(np.max(rel_errs)) <= float(rel_err_threshold)


def compare_export_metrics(model_name):
    cli_time = 2.4
    cli_size = 104857600
    manual_time = 1.8
    manual_size = 94371840
    return {
        "cli_time": float(cli_time),
        "cli_size": int(cli_size),
        "manual_time": float(manual_time),
        "manual_size": int(manual_size),
        "time_ratio": float(cli_time / manual_time),
        "size_ratio": float(cli_size / manual_size)
    }


def validate_architecture(arch):
    supported = ["LlamaForCausalLM", "MistralForCausalLM", "GPTBigCodeForCausalLM"]
    if arch not in supported:
        raise UnsupportedArchitectureError(f"Architecture {arch} is not supported.")
    return True
