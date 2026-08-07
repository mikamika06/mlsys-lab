def annotate_phases(trace_data):
    annotated = []
    for item in trace_data:
        p = dict(item)
        p["annotated"] = True
        annotated.append(p)
    return annotated
