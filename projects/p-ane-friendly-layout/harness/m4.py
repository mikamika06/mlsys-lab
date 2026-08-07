import ref
from ane_model.transform import make_ane_friendly, verify_parity

def check(workdir):
    m = {"max_diff": 1.0}
    model = ref.get_sample_model()
    t_model = make_ane_friendly(model.copy())
    inp = ref.get_sample_input()
    diff = verify_parity(model, t_model, inp)
    m["max_diff"] = float(diff)
    return m
