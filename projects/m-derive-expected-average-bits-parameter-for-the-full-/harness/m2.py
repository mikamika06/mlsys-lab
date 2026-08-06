import ref
from qlora.quant import quantize_and_measure


def check(workdir):
    out = {"error_measured": 0.0}
    t = ref.get_test_tensor()
    mse, rec = quantize_and_measure(t, quant_type="nf4")
    ref_mse, _ = ref.quantize_and_measure(t, quant_type="nf4")
    if abs(mse - ref_mse) < 1e-5 and rec.shape == t.shape:
        out["error_measured"] = 1.0
    else:
        out["_note"] = f"mse {mse} differs from ref {ref_mse}"
    return out
