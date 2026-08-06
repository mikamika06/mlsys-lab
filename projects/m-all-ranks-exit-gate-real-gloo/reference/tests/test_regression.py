import os
import shutil
import tempfile
from ft.checkpoint import get_safe_resume_checkpoint

def test_corrupted_latest_pointer():
    d = tempfile.mkdtemp()
    try:
        step1 = os.path.join(d, "step_1")
        os.makedirs(step1)
        open(os.path.join(step1, "model.pt"), "w").close()

        step2 = os.path.join(d, "step_2")
        os.makedirs(step2)

        with open(os.path.join(d, "latest"), "w") as f:
            f.write("step_2")
        
        res = get_safe_resume_checkpoint(d)
        assert res == step1, f"Expected {step1}, got {res}"
    finally:
        shutil.rmtree(d)
