import sys
sys.path.insert(0, ".")


def test_dequantize_uses_absmax():
    """
    Write a test that verifies `dequantize_blockwise` properly scales values
    by the block's `absmax`. The test must fail if the implementation incorrectly
    ignores `absmax` and merely returns unscaled codebook lookups.
    """
    raise NotImplementedError()
