import sys
sys.path.insert(0, ".")
from qfusion.fusion import predict_fusion
from qfusion.placement import insert_per_channel_qdq
from qfusion.engine import evaluate_engine

def test_predict_fusion_basic():
    node = {"op": "MatMul", "has_scale": True, "axis": 0}
    assert predict_fusion(node) == "fusion"

def test_insert_per_channel_qdq_axis():
    info = {"name": "w", "shape": (128, 64), "axis": 0}
    res = insert_per_channel_qdq(info)
    assert res["axis"] == 0
    assert res["scale_shape"] == (128,)

def test_evaluate_engine_selection():
    assert evaluate_engine(10.0, 15.0) == "int8"
    assert evaluate_engine(14.0, 15.0) == "fp16"
