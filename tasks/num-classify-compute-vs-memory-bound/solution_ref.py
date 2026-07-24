def classify_roofline_ops(ops, peak_flops, bandwidth):
    balance = peak_flops / bandwidth
    result = []
    for op in ops:
        ai = float(op["flops"]) / float(op["bytes"])
        bound = "compute" if ai >= balance else "memory"
        result.append(
            {
                "name": op["name"],
                "ai": ai,
                "bound": bound,
            }
        )
    return result
