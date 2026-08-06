import os
import json
import ref


def check(workdir):
    import trtexplore.analyze as ta
    
    raw, simp, _, _ = ref.generate_fixtures()
    raw_path = os.path.join(workdir, "raw.json")
    simp_path = os.path.join(workdir, "simp.json")
    
    with open(raw_path, "w") as f:
        json.dump(raw, f)
    with open(simp_path, "w") as f:
        json.dump(simp, f)
        
    out = {"metrics_match": 0.0}
    try:
        got = ta.analyze_engine(raw_path, simp_path)
        want = ref.analyze_engine(raw_path, simp_path)
        if got == want:
            out["metrics_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"crashed: {e}"
        
    return out
