import ref

def check(workdir):
    m = {"metrics_correct": 0.0}
    try:
        from oom_triage.analyzer import analyze_fragmentation
        snap = {
            "segments": [
                {
                    "size": 100,
                    "blocks": [
                        {"id": 1, "size": 30, "state": "allocated"},
                        {"id": 0, "size": 20, "state": "free"},
                        {"id": 2, "size": 10, "state": "allocated"},
                        {"id": 0, "size": 40, "state": "free"}
                    ]
                }
            ]
        }
        res = analyze_fragmentation(snap)
        if res.get("allocated") == 40 and res.get("reserved") == 100 and res.get("max_free") == 40:
            m["metrics_correct"] = 1.0
    except Exception:
        pass
    return m
