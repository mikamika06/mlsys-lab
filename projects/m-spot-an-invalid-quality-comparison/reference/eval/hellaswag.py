import math


def run_hellaswag_eval(items, model_fn):
    if not items:
        return {"acc": 0.0, "stderr": 0.0, "count": 0}
    correct_list = []
    for item in items:
        ctx = item["context"]
        endings = item["endings"]
        label = item["label"]
        scores = model_fn(ctx, endings)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct_list.append(1.0 if pred == label else 0.0)
    n = len(correct_list)
    acc = sum(correct_list) / n
    if n <= 1:
        stderr = 0.0
    else:
        var = sum((x - acc) ** 2 for x in correct_list) / (n - 1)
        stderr = math.sqrt(var / n)
    return {"acc": float(acc), "stderr": float(stderr), "count": n}
