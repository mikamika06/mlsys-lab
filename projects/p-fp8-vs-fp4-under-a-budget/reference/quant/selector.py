def check_kernel_support(format_name, architecture):
    support_matrix = {
        "hopper": {"fp8": True, "fp6": False, "fp4": False, "int4": True},
        "blackwell": {"fp8": True, "fp6": True, "fp4": True, "int4": True}
    }
    return support_matrix.get(architecture, {}).get(format_name, False)

def recommend_format(candidates, budget_bpw, architecture):
    best_fmt = None
    min_error = float('inf')
    for fmt, bpw, error in candidates:
        if bpw <= budget_bpw:
            supported = check_kernel_support(fmt, architecture)
            if supported and error < min_error:
                min_error = error
                best_fmt = fmt
    if best_fmt is None:
        for fmt, bpw, error in candidates:
            if bpw <= budget_bpw and error < min_error:
                min_error = error
                best_fmt = fmt
    return best_fmt or "fp8"
