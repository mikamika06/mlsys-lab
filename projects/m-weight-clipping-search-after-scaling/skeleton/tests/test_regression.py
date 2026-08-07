import sys

sys.path.insert(0, ".")
from awq_clip.quant import search_clipping, quantize_and_reconstruct


def test_clipping_never_worse_than_unclipped():
    raise NotImplementedError
