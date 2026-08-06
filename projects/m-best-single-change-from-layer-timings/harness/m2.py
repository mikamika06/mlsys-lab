import os
import json
import ref


def check(workdir):
    import trtexplore.analyze as ta
    
    _, _, prof, candidates = ref.generate_fixtures()
    prof_path = os.path.join(workdir, "prof.json")
    
    with open(prof_path, "w") as f:
        json.dump(prof, f)
        
    out = {"argmin_index": 0.0}
    try:
        got = ta.best_single_change(prof_path, candidates)
        want = ref.best_single_change(prof_path, candidates)
        if got == want:
            out["argmin_index"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"crashed: {e}"
        
    return out
