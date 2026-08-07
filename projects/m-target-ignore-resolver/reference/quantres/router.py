def find_wrong_router(quantized_modules):
    wrong = []
    for name, dtype in quantized_modules.items():
        if ("gate" in name or "router" in name) and dtype != "fp32":
            wrong.append(name)
    return sorted(wrong)
