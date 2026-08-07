import ref
from ane_model.audit import find_fallback_ops

def check(workdir):
    m = {"fallbacks_found": 0.0}
    model = ref.get_sample_model()
    ops = find_fallback_ops(model)
    if isinstance(ops, list) and len(ops) >= 2:
        m["fallbacks_found"] = float(len(ops))
    return m
