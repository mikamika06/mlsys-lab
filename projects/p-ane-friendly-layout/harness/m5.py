import ref
from ane_model.transform import make_ane_friendly, measure_ane_fraction

def check(workdir):
    m = {"ane_fraction": 0.0}
    model = ref.get_sample_model()
    t_model = make_ane_friendly(model.copy())
    inp = ref.get_sample_input()
    frac = measure_ane_fraction(t_model, inp)
    m["ane_fraction"] = float(frac)
    return m
