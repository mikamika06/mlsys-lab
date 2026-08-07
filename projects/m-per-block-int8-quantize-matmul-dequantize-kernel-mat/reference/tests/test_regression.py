import sys
sys.path.insert(0, ".")
import numpy as np
from qmat.matmul import per_block_int8_matmul

def test_block_quantization_preserves_cross_block_signals():
    """Ensure per-block quantization isolates outliers properly."""
    A = np.array([[1e6, 1.0, 1.0, 1.0]], dtype=np.float32)
    B = np.array([[1.0], [1.0], [1e6], [1e6]], dtype=np.float32)

    C_ref = np.dot(A, B)
    C_q = per_block_int8_matmul(A, B, block_size=2)

    rel_err = float(np.abs(C_ref - C_q)[0, 0] / C_ref[0, 0])
    assert rel_err < 0.1, f"Quantization error too large: {rel_err}"
