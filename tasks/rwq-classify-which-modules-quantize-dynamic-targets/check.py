def _oracle(names):
    quantizable = {"Linear", "LSTM"}
    return [1 if n in quantizable else 0 for n in names]


def grade(sol, fx) -> dict:
    cases = [
        ["Linear", "Conv2d", "LSTM"],
        ["Embedding", "BatchNorm2d", "Linear"],
        [],
        ["LSTM", "LSTM", "Linear", "Dropout"],
        ["linear", "LSTM", "LINEAR"]
    ]
    for names in cases:
        try:
            got = sol.classify_quantizable(names)
        except Exception:
            return {"exact_match": 0.0}
        if list(got) != _oracle(names):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
