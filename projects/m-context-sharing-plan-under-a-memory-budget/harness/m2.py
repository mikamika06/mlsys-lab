import ref


def check(workdir):
    from ctxplan.classifier import classify_oom
    logs = [
        {"phase": "build", "peak_memory": 1000},
        {"phase": "runtime", "peak_memory": 8000},
        {"phase": "runtime", "peak_memory": 2000}
    ]
    workspace_limit = 5000
    expected = ["build", "build", "runtime"]
    correct = 0
    for log, exp in zip(logs, expected):
        try:
            res = classify_oom(log, workspace_limit)
            if res == exp:
                correct += 1
        except Exception:
            pass
    out = {"classification_accuracy": 1.0 if correct == len(expected) else 0.0}
    return out
