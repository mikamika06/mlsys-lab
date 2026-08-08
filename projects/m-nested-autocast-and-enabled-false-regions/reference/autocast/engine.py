def simulate_stack(events):
    stack = []
    results = []
    for ev in events:
        t = ev.get("type")
        if t == "push":
            stack.append({"device": ev.get("device", "cuda"), "dtype": ev.get("dtype", "float16"), "enabled": ev.get("enabled", True)})
        elif t == "pop":
            if stack:
                stack.pop()
        current = dict(stack[-1]) if stack else {"device": None, "dtype": None, "enabled": False}
        results.append(current)
    return results
