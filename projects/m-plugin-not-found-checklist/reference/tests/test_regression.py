import sys
sys.path.insert(0, ".")

from trtplugin.serialize import serialize_fields, deserialize_fields
from trtplugin.decision import decide_op_strategy


def test_field_serialization_roundtrip():
    payload = {
        "alpha": 0.125,
        "clip_max": 6,
        "dims": [1, 3, 224, 224],
        "op_mode": "fast_math"
    }
    encoded = serialize_fields(payload)
    decoded = deserialize_fields(encoded)
    assert decoded == payload, f"Deserialized payload {decoded} does not match {payload}"


def test_op_strategy_native_priority():
    node = {"op_type": "Relu", "has_custom_kernel": False, "decomposable_to_native": False}
    strategy = decide_op_strategy(node, trt_native_ops={"Relu"}, available_plugins=["ReluPlugin"], constraints={"allow_plugin": True, "perf_critical": True})
    assert strategy == "NATIVE", f"Expected NATIVE, got {strategy}"


def test_op_strategy_fallback():
    node = {"op_type": "UnknownCustomOp", "has_custom_kernel": False, "decomposable_to_native": False}
    strategy = decide_op_strategy(node, trt_native_ops={"Conv"}, available_plugins=[], constraints={"allow_plugin": False, "perf_critical": False})
    assert strategy == "FALLBACK", f"Expected FALLBACK, got {strategy}"
