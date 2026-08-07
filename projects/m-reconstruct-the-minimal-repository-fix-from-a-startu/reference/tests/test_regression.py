import sys
sys.path.insert(0, ".")
from tritonfix.reconstruct import reconstruct_fix
from tritonfix.detect import detect_mismatch

def test_reconstruct_fix_non_empty():
    logs = ["I0801 model_repository_manager.cc:123] Poll failed for model 'test_mod': version 1 is missing file 'model.onnx'"]
    fixes = reconstruct_fix(logs)
    assert len(fixes) > 0
    assert fixes[0]["model"] == "test_mod"

def test_detect_mismatch_onnx():
    config = 'backend: "onnxruntime"'
    files = ["model.pt"]
    assert detect_mismatch(config, files) is True
