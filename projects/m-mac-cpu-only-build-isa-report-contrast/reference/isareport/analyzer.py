def has_mismatch(native_rep, manual_rep):
    from isareport.contrast import contrast_isa
    return len(contrast_isa(native_rep, manual_rep)) > 0

def analyze_tier(features):
    if features.get("fp16") and features.get("neon"):
        return "T1"
    return "T0"
