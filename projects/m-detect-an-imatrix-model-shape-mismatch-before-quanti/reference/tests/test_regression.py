from imatrix.validator import validate_imatrix

def test_shape_mismatch_detection():
    model_tensors = {"layer.0.weight": [1024, 1024]}
    imatrix_data = {"layer.0.weight": {"shape": [512, 1024]}}
    res = validate_imatrix(model_tensors, imatrix_data)
    assert res["valid"] is False
    assert len(res["mismatches"]) == 1

def test_missing_tensor_detection():
    model_tensors = {"layer.0.weight": [1024, 1024]}
    imatrix_data = {}
    res = validate_imatrix(model_tensors, imatrix_data)
    assert res["valid"] is False
    assert len(res["missing"]) == 1

def test_valid_imatrix_passes():
    model_tensors = {"layer.0.weight": [1024, 1024]}
    imatrix_data = {"layer.0.weight": {"shape": [1024, 1024]}}
    res = validate_imatrix(model_tensors, imatrix_data)
    assert res["valid"] is True
    assert len(res["mismatches"]) == 0
    assert len(res["missing"]) == 0
