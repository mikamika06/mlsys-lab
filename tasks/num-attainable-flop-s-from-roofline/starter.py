import numpy as np


def roofline_attainable(flops, bytes_moved, peak_flops, bandwidth):
    """Return (ai, attainable, ridge) for the roofline model.

    ai         -- flops / bytes_moved, in FLOP/byte
    attainable -- min(peak_flops, ai * bandwidth), in FLOP/s
    ridge      -- peak_flops / bandwidth, in FLOP/byte
    """
    raise NotImplementedError('your code here')
