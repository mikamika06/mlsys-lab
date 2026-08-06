import ref


def check(workdir):
    from troubleshoot.diagnosis import diagnose_errors

    items = ref.get_error_items()
    transcripts = [it["transcript"] for it in items]
    logs = [f"log for {it['transcript']}" for it in items]
    ps_list = [{"rss": 1024} for _ in items]
    mem_list = [{"free": 4096} for _ in items]

    try:
        results = diagnose_errors(transcripts, logs, ps_list, mem_list)
    except Exception as e:
        return {"diagnoses_correct": 0.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}

    if not isinstance(results, (list, dict)):
        return {"diagnoses_correct": 0.0, "_note": "diagnose_errors must return list or dict"}

    correct = 0
    for i, it in enumerate(items):
        got = results[i] if isinstance(results, list) else results.get(it["transcript"])
        if isinstance(got, dict) and got.get("root_cause") == it["root_cause"] and got.get("fix") == it["fix"]:
            correct += 1
    return {"diagnoses_correct": float(correct)}
