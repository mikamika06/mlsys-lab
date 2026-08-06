import ref


def check(workdir):
    from troubleshoot.mapper import map_transcripts

    items = ref.get_error_items()
    transcripts = [it["transcript"] for it in items]
    try:
        results = map_transcripts(transcripts)
    except Exception as e:
        return {"mapped_correctly": 0.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}

    if not isinstance(results, (list, dict)):
        return {"mapped_correctly": 0.0, "_note": "map_transcripts must return list or dict"}

    correct = 0
    for i, it in enumerate(items):
        want = it["failure_class"]
        got = results[i] if isinstance(results, list) else results.get(it["transcript"])
        if got == want:
            correct += 1
    return {"mapped_correctly": float(correct)}
