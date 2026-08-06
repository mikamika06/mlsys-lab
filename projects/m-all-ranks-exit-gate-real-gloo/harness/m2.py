import os
import tempfile
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from ft.checkpoint import get_safe_resume_checkpoint
    except ImportError:
        return {"_note": "failed to import"}

    out = {"valid_latest": 0.0, "corrupted_latest": 0.0, "missing_latest": 0.0}

    with tempfile.TemporaryDirectory() as d:
        step1 = os.path.join(d, "step_1")
        os.makedirs(step1)
        open(os.path.join(step1, "model.pt"), "w").close()
        with open(os.path.join(d, "latest"), "w") as f:
            f.write("step_1")
            
        res = get_safe_resume_checkpoint(d)
        if res == step1:
            out["valid_latest"] = 1.0

    with tempfile.TemporaryDirectory() as d:
        step2 = os.path.join(d, "step_2")
        os.makedirs(step2)
        open(os.path.join(step2, "model.pt"), "w").close()
        
        step3 = os.path.join(d, "step_3")
        os.makedirs(step3)
        
        with open(os.path.join(d, "latest"), "w") as f:
            f.write("step_3")
            
        res = get_safe_resume_checkpoint(d)
        if res == step2:
            out["corrupted_latest"] = 1.0
            
    with tempfile.TemporaryDirectory() as d:
        step1 = os.path.join(d, "step_1")
        os.makedirs(step1)
        open(os.path.join(step1, "model.pt"), "w").close()
        
        step4 = os.path.join(d, "step_4")
        os.makedirs(step4)
        open(os.path.join(step4, "model.pt"), "w").close()
        
        res = get_safe_resume_checkpoint(d)
        if res == step4:
            out["missing_latest"] = 1.0

    return out
