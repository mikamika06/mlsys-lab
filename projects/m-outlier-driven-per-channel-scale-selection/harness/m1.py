import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import quant.scale as qs

        out = {"max_scale_match": 0.0, "sim_match": 0.0}
        chan = ref.W_TEST[0]

        got_scale = qs.compute_max_scale(chan)
        want_scale = ref.compute_max_scale(chan)
        if np.isclose(got_scale, want_scale):
            out["max_scale_match"] = 1.0

        got_sim = qs.simulate_quant(chan, got_scale)
        want_sim = ref.simulate_quant(chan, want_scale)
        if np.allclose(got_sim, want_sim):
            out["sim_match"] = 1.0

        return out
    finally:
        sys.path.pop(0)
