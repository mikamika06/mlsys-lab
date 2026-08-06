import sys
import numpy as np
sys.path.insert(0, ".")
import ref
from ortinfer.session import run_inference, get_optimized_node_count


def test_output_parity_against_reference():
    model_bytes = ref.get_model_bytes()
    x = np.array([[0.5, -1.5, 2.5, -3.5]], dtype=np.float32)
    got = run_inference(model_bytes, x)
    want = ref.reference_inference(x)
    assert np.allclose(got, want, atol=1e-5)


def test_optimization_level_nodes():
    model_bytes = ref.get_model_bytes()
    basic_cnt = get_optimized_node_count(model_bytes, "BASIC")
    all_cnt = get_optimized_node_count(model_bytes, "ALL")
    assert basic_cnt >= all_cnt
