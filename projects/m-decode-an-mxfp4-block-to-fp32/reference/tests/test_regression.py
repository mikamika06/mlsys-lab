import numpy as np
from mxfp4.decode import decode_mxfp4_block


def test_mxfp4_scale_invariance():
    nibbles = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 2, dtype=np.uint8)
    dec1 = decode_mxfp4_block(127, nibbles)
    dec2 = decode_mxfp4_block(128, nibbles)

    assert dec1[2] == 1.0
    assert dec2[2] == 2.0
    assert np.allclose(dec2, dec1 * 2.0)
