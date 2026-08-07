import ref
from ane_model.transform import make_ane_friendly

def check(workdir):
    m = {"transformed_ok": 0.0}
    model = ref.get_sample_model()
    t_model = make_ane_friendly(model)
    if t_model.get("optimized", False):
        m["transformed_ok"] = 1.0
    return m
