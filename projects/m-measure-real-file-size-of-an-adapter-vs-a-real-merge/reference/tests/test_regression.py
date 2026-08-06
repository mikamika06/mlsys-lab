import sys
import numpy as np

sys.path.insert(0, ".")
from adaptermerge.measure import measure_file_sizes
from adaptermerge.quantize import compute_quantization_error
from adaptermerge.unload import merge_and_unload


def test_merge_and_unload_removes_lora_keys():
    state = {
        "layer.weight": np.zeros((4, 4)),
        "layer.lora_A": np.zeros((2, 4)),
        "layer.lora_B": np.zeros((4, 2))
    }
    unloaded = merge_and_unload(state)
    for k in unloaded.keys():
        assert "lora" not in k, f"Leaked lora key {k}"
        assert "adapter" not in k, f"Leaked adapter key {k}"


def test_measure_file_sizes_returns_ratio():
    base = {"w": np.ones((10, 10))}
    adapter = {"a": np.ones((10, 10))}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        res = measure_file_sizes(base, adapter, tmp)
        assert "size_ratio" in res
        assert res["size_ratio"] > 0


def test_quantization_error_structure():
    base = {"layer.weight": np.ones((10, 10))}
    adapter = {"layer.lora_A": np.ones((2, 10)), "layer.lora_B": np.ones((10, 2))}
    res = compute_quantization_error(base, adapter)
    assert "error_merged" in res
