import numpy as np
import ref


def check(workdir):
    m = {"kld_bounded": 0.0, "ppl_sensible": 0.0}
    import sys
    sys.path.insert(0, workdir)
    try:
        import gguf_pipe.eval as ev
        r1 = np.array([1.0, 2.0, 3.0])
        r2 = np.array([1.1, 1.9, 3.1])
        kld = ev.compute_kld(r1, r2)
        if kld >= 0.0 and kld < 1.0:
            m["kld_bounded"] = 1.0
        ppl = ev.compute_perplexity("model_Q4_K_M.gguf", ["test"])
        if 0.0 < ppl < 100.0:
            m["ppl_sensible"] = 1.0
    except Exception:
        pass
    return m
