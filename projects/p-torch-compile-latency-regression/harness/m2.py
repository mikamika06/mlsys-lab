import importlib.util
import os

import w


def check(workdir):
    out = {"has_inventory": 0.0, "graph_count_ok": 0.0, "break_count_ok": 0.0, "op_count_ok": 0.0}
    torch = w.torch_or_none()
    if torch is None:
        return w.needs_torch(out)
    path = os.path.join(workdir, "tools", "inventory.py")
    if not os.path.isfile(path):
        out["_note"] = "tools/inventory.py is missing"
        return out
    spec = importlib.util.spec_from_file_location("learner_inventory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out["has_inventory"] = 1.0

    import torch._dynamo as dyn
    Model, norm = w.baseline()
    torch.manual_seed(0)
    m = Model().eval()
    x = norm(torch.randn(4, 64))

    dyn.reset()
    got = mod.inventory(m, x)
    dyn.reset()
    exp = dyn.explain(m, x)

    out["graph_count_ok"] = 1.0 if got.get("graph_count") == exp.graph_count else 0.0
    out["break_count_ok"] = 1.0 if got.get("graph_break_count") == exp.graph_break_count else 0.0
    out["op_count_ok"] = 1.0 if got.get("op_count") == exp.op_count else 0.0
    if not all((out["graph_count_ok"], out["break_count_ok"], out["op_count_ok"])):
        out["_note"] = (f"got {got}, reference "
                        f"{{'graph_count': {exp.graph_count}, 'graph_break_count': "
                        f"{exp.graph_break_count}, 'op_count': {exp.op_count}}}")
    return out
