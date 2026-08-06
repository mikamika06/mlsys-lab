import torch
from qlora.quant import quantize_and_measure


def compare_nf4_fp4(tensor):
    err_nf4, _ = quantize_and_measure(tensor, quant_type="nf4")
    err_fp4, _ = quantize_and_measure(tensor, quant_type="fp4")
    return {"nf4_mse": float(err_nf4), "fp4_mse": float(err_fp4), "better": "nf4" if err_nf4 <= err_fp4 else "fp4"}
