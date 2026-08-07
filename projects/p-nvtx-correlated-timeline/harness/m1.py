import ref

def check(workdir):
    m = {"phases_annotated": 0.0}
    try:
        from profiler.annotations import annotate_phases
        data = ref.get_sample_trace()
        res = annotate_phases(data)
        if isinstance(res, list) and len(res) > 0 and all("annotated" in x for x in res):
            m["phases_annotated"] = 1.0
    except Exception:
        pass
    return m
