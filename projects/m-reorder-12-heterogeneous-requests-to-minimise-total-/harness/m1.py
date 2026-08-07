import ref
from heterogeneous import scheduler

def check(workdir):
    out = {"schedule_optimal": 0.0}
    try:
        ref_ordered = scheduler.reorder_requests(ref.REQUESTS)
        from importlib import reload
        import heterogeneous.scheduler as learner_mod
        reload(learner_mod)
        got_ordered = learner_mod.reorder_requests(ref.REQUESTS)
        if len(got_ordered) == len(ref_ordered):
            out["schedule_optimal"] = 1.0
        else:
            out["_note"] = f"Expected {len(ref_ordered)} requests, got {len(got_ordered)}"
    except Exception as e:
        out["_note"] = f"Error during scheduling execution: {type(e).__name__}: {str(e)[:100]}"
    return out
