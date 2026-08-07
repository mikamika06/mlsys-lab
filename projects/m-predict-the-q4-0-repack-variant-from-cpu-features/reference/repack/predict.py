def predict_variant(features):
    if features.get("sve", False):
        return "q4_0_sve"
    if features.get("avx512f", False):
        return "q4_0_avx512"
    if features.get("avx2", False):
        return "q4_0_avx2"
    if features.get("neon", False):
        return "q4_0_neon"
    return "q4_0_scalar"
