def generate_summary(bias, comparison):
    status = "biased" if bias > 0.1 else "stable"
    return {"status": status, "bias": bias, "comparison": comparison}
