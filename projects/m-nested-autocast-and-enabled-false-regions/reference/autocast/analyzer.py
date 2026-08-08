def analyze_regions(trace):
    stack = []
    violations = 0
    correct_count = 0
    for item in trace:
        ev = item.get("event")
        if ev == "push":
            stack.append({"enabled": item.get("enabled", True), "dtype": item.get("dtype", "float16")})
        elif ev == "pop":
            if stack:
                stack.pop()
        elif ev == "op":
            current_enabled = stack[-1]["enabled"] if stack else False
            op_requires_fp32 = item.get("sensitive", False)
            if op_requires_fp32 and current_enabled and item.get("used_dtype") != "float32":
                violations += 1
            elif not current_enabled and item.get("used_dtype") == "float16":
                violations += 1
            else:
                correct_count += 1
    return {"violations": violations, "correct": correct_count}
