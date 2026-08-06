import sys
sys.path.insert(0, ".")
from ctinspect.repair import repair_missing_weight_shape
from ctinspect.calc import calculate_quant_byte_size

SAMPLE_INDEX = {
    "weight_map": {
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
        "model.layers.0.self_attn.q_proj.weight_scale": "model-00001-of-00001.safetensors"
    },
    "tensor_metadata": {
        "model.layers.0.self_attn.q_proj.weight": {
            "bits": 4,
            "packed_byte_size": 2048,
            "scale_shape": [64, 1]
        }
    }
}

def test_repaired_weight_shapes_match_scales():
    repaired = repair_missing_weight_shape(SAMPLE_INDEX)
    meta = repaired["tensor_metadata"]["model.layers.0.self_attn.q_proj.weight"]
    assert "weight_shape" in meta
    shape = meta["weight_shape"]
    scale_shape = meta["scale_shape"]
    assert shape[0] == scale_shape[0]

def test_packed_vs_int_byte_sizes():
    shape = [64, 64]
    int_bytes = calculate_quant_byte_size(shape, num_bits=4, packed=False)
    pack_bytes = calculate_quant_byte_size(shape, num_bits=4, packed=True)
    assert pack_bytes < int_bytes
    assert pack_bytes == 2048
    assert int_bytes == 4096
