import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = [
    ["tensor 'L['x']' size mismatch at index 0. expected 1, actual 2"],
    ["tensor 'L['x']' size mismatch at index 0. expected 2, actual 3",
     "tensor 'L['x']' size mismatch at index 0. expected 1, actual 3"],
    ["L['self'].scale == 1.0"],
    ["tensor 'L['x']' dtype mismatch. expected torch.float32, actual torch.float64"],
]


def check(workdir):
    out = {"has_module": 0.0, "events_found": 0.0, "guards_match": 0.0}
    path = os.path.join(workdir, "tools", "guards.py")
    if not os.path.isfile(path):
        out["_note"] = "tools/guards.py is missing"
        return out
    spec = importlib.util.spec_from_file_location("learner_guards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out["has_module"] = 1.0

    with open(os.path.join(HERE, "fixtures_recompiles.txt"), encoding="utf-8") as f:
        text = f.read()
    got = mod.failing_guards(text)
    out["events_found"] = 1.0 if len(got) == len(EXPECTED) else 0.0
    norm = [[g.strip() for g in ev] for ev in got] if isinstance(got, list) else []
    out["guards_match"] = 1.0 if norm == EXPECTED else 0.0
    if not out["guards_match"]:
        out["_note"] = f"got {norm[:2]}..., expected {EXPECTED[:2]}..."
    return out
