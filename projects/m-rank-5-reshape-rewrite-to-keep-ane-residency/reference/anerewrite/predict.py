def predict_ane_friendly(arch_a, arch_b):
    score_a = -arch_a["max_rank"] * 10 - arch_a["reshapes"]
    score_b = -arch_b["max_rank"] * 10 - arch_b["reshapes"]
    if score_a >= score_b:
        return "arch_a"
    return "arch_b"
