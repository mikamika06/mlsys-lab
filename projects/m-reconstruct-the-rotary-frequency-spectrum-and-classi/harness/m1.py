import ref
import numpy as np

def check(workdir):
    from ropespectrum.spectrum import reconstruct_spectrum
    out = {"spectrum_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.reconstruct_spectrum(cfg["dim"], cfg["base"])
        try:
            got = reconstruct_spectrum(cfg["dim"], cfg["base"])
            if got is not None and np.allclose(got, want, rtol=1e-5, atol=1e-5):
                ok += 1
        except Exception:
            pass
    out["spectrum_matched"] = float(ok)
    return out
