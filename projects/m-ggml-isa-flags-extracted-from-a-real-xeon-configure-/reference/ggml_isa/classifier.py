def classify_format(fmt_name):
    f = fmt_name.upper()
    if "AMX" in f or "Q8_0" in f or "Q5_K" in f:
        return "accelerated"
    if "Q4_0" in f:
        return "standard"
    return "fallback"
