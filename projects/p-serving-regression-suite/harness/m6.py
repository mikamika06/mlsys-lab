import ref
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"false_positives": 3.0, "safeguard_caught": 0.0}

    if ref.run_tests(workdir) is not True:
        return out

    fps = 0
    for _ in range(3):
        if not ref.survives(workdir):
            fps += 1
    out["false_positives"] = float(fps)

    import serving.engine as eng
    good_gen = eng.Engine.generate

    def catastrophic(self, prompt, **kwargs):
        return {}

    eng.Engine.generate = catastrophic
    try:
        if not ref.survives(workdir):
            out["safeguard_caught"] = 1.0
    finally:
        eng.Engine.generate = good_gen

    return out
