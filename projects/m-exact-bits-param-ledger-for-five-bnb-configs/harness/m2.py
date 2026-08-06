import ref
import numpy as np

def check(workdir):
    from bnb_ledger.quant import nested_absmax_quantize
    np.random.seed(42)
    x = np.random.randn(1024)
    want = ref.nested_absmax_quantize(x, block_size=256)
    got = nested_absmax_quantize(x, block_size=256)
    err = float(np.mean(np.abs(want - got)))
    out = {"rel_err": err}
    return out
