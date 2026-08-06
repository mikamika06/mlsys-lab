import ref


def check(workdir):
    from ollama_evict.tracker import ModelTracker
    out = {"tracker_match": 0.0}
    try:
        ok = True
        for scn in ref.get_test_scenarios():
            t = ModelTracker()
            ref_t = ref.ReferenceTracker()
            t_time = 0
            for req in scn["requests"]:
                t_time += 1
                t.touch(req, t_time)
                ref_t.touch(req, t_time)
            if t.states != ref_t.states:
                ok = False
        out["tracker_match"] = 1.0 if ok else 0.0
    except Exception as e:
        out["_note"] = f"error in m1: {type(e).__name__}: {str(e)[:120]}"
    return out
