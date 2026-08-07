import ref
import numpy as np

def check(workdir):
    from snrchk.model import BundledProgram, run_eager
    from snrchk.analyzer import compute_layer_snrs

    out = {"snr_match": 0.0}
    np.random.seed(100)
    x = np.random.randn(4, 16)
    for case in ref.TEST_CASES:
        weights = ref.make_model(case["num_layers"], seed=123)
        eager = run_eager(weights, x)
        prog = BundledProgram(weights, diverge_layer=case["diverge_layer"])
        exported = prog.run_exported(x)
        got_snrs = compute_layer_snrs(eager, exported)
        want_snrs = ref.compute_snr(eager, exported)
        if not np.allclose(got_snrs, want_snrs, atol=1e-3):
            out["_note"] = f"SNR mismatch: got {got_snrs}, want {want_snrs}"
            return out
    out["snr_match"] = 1.0
    return out
