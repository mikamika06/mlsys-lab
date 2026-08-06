def classify_run(run):
    if run["threads_allocated"] <= run["physical_cores"] and run["contention_score"] < 0.5:
        return "fair"
    return "unfair"


def classify_runs(runs):
    return [classify_run(r) for r in runs]
