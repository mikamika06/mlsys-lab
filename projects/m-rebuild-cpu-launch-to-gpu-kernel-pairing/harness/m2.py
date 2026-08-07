import ref

def check(workdir):
    from trace_parser.classify import classify_slices
    events = ref.generate_trace()
    want = ref.classify_slices(events)
    try:
        got = classify_slices(events)
    except Exception:
        return {"classification_accuracy": 0.0}
    if got == want:
        return {"classification_accuracy": 1.0}
    return {"classification_accuracy": 0.0}
