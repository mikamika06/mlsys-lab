import ref
import numpy as np

def check(workdir):
    from exporter.core import translate_branches

    m = {"primitive_translated": 0.0}
    x = np.array([0.2, 0.8])
    out = translate_branches(x)
    expected = np.where(x > 0.5, x * 2.0, x + 1.0)
    if np.allclose(out, expected):
        m["primitive_translated"] = 1.0
    return m
