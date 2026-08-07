import ref
import numpy as np

def check(workdir):
    from snrchk.model import BundledProgram, run_eager
    from snrchk.analyzer import compute_layer_snrs, bisect_divergence

    out = {"bisect_match": 0.0}
    np.random.seed(200)
    x = np.random.randn(4, 16)
    for case in ref.TEST_CASES:
        weights = ref.make_model(case["num_layers"], seed=456)
        eager = run_eager(weights, x)
        prog = BundledProgram(weights, diverge_layer=case["diverge_layer"])
        exported = prog.run_exported(x)
        snrs = compute_layer_snrs(eager, exported)
        got_layer = bisect_divergence(snrs, threshold=case["threshold"])
        want_layer = ref.find_diverging_layer(snrs, threshold=case["threshold"])
        if got_layer != want_layer:
            out["_note"] = f"Bisection mismatch: got layer {got_layer}, want {want_layer}"
            return out
    out["bisect_match"] = 1.0
    return out
