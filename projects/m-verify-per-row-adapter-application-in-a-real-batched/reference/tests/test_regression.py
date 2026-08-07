import sys
import numpy as np

sys.path.insert(0, ".")
from loraserving.routing import apply_per_row_lora, verify_batched_lora


def test_row_indexing_integrity():
    np.random.seed(42)
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    adapter_ids = np.array([0, 1, 0], dtype=np.int32)
    lora_a = np.ones((2, 2, 2), dtype=np.float32)
    lora_b = np.ones((2, 2, 2), dtype=np.float32)
    scaling = np.array([1.0, 2.0], dtype=np.float32)

    res = apply_per_row_lora(x, adapter_ids, lora_a, lora_b, scaling)

    row0 = np.dot(np.dot(x[0], lora_a[0]), lora_b[0]) * scaling[0]
    row1 = np.dot(np.dot(x[1], lora_a[1]), lora_b[1]) * scaling[1]
    row2 = np.dot(np.dot(x[2], lora_a[0]), lora_b[0]) * scaling[0]

    expected = np.vstack([row0, row1, row2])
    v = verify_batched_lora(x, adapter_ids, lora_a, lora_b, scaling, expected)
    assert v["is_correct"], f"Routing integrity check failed: max error {v['max_error']}"


def test_unassigned_adapter_pass_through():
    x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    adapter_ids = np.array([-1, 0], dtype=np.int32)
    lora_a = np.ones((1, 2, 2), dtype=np.float32)
    lora_b = np.ones((1, 2, 2), dtype=np.float32)
    scaling = 1.0

    out = apply_per_row_lora(x, adapter_ids, lora_a, lora_b, scaling)
    assert np.allclose(out[0], 0.0), "Row with adapter_id -1 must produce zero adapter output"
    assert not np.allclose(out[1], 0.0), "Row with valid adapter_id must compute adapter output"
