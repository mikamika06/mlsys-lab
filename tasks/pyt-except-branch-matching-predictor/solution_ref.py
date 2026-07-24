def predict_except_star(names):
    remaining = list(names)
    fired = []

    if any(name == "ValueError" for name in remaining):
        fired.append(0)
        remaining = [name for name in remaining if name != "ValueError"]

    if any(name in ("KeyError", "IndexError") for name in remaining):
        fired.append(1)
        remaining = [
            name for name in remaining
            if name not in ("KeyError", "IndexError")
        ]

    if any(name == "TypeError" for name in remaining):
        fired.append(2)
        remaining = [name for name in remaining if name != "TypeError"]

    if remaining:
        fired.append(3)

    return fired
