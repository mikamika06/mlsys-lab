import math

INDEX_FIXTURES = [
    {
        "metadata": {"quantization_config": {"format": "pack-quantized"}},
        "weight_map": {
            "layer.0.packed_weight": "file1.safetensors",
            "layer.0.weight_scale": "file1.safetensors"
        },
        "tensor_metadata": {
            "layer.0.weight": {"bits": 4, "packed_byte_size": 1024, "scale_shape": [32, 1]}
        }
    },
    {
        "metadata": {"quantization_config": {"format": "int-quantized"}},
        "weight_map": {
            "layer.0.weight": "file1.safetensors",
            "layer.0.weight_scale": "file1.safetensors"
        },
        "tensor_metadata": {
            "layer.0.weight": {"bits": 8, "packed_byte_size": 4096, "scale_shape": [64, 1]}
        }
    },
    {
        "metadata": {},
        "weight_map": {
            "layer.0.weight_packed": "file1.safetensors",
            "layer.0.scale": "file1.safetensors"
        },
        "tensor_metadata": {
            "layer.0.weight": {"bits": 2, "packed_byte_size": 512, "scale_shape": [16, 1]}
        }
    },
    {
        "metadata": {},
        "weight_map": {
            "layer.0.weight": "file1.safetensors",
            "layer.0.zero_point": "file1.safetensors"
        },
        "tensor_metadata": {
            "layer.0.weight": {"bits": 8, "packed_byte_size": 2048, "scale_shape": [32, 1]}
        }
    },
    {
        "metadata": {},
        "weight_map": {
            "layer.0.weight": "file1.safetensors"
        },
        "tensor_metadata": {}
    }
]

def infer_scheme_from_index(index_data):
    weight_map = index_data.get("weight_map", {})
    keys = list(weight_map.keys())
    has_packed = any(".weight_packed" in k or ".packed_weight" in k for k in keys)
    has_scale = any(".weight_scale" in k or ".scale" in k for k in keys)

    metadata = index_data.get("metadata", {})
    quant_config = metadata.get("quantization_config", {})
    format_str = quant_config.get("format", "").lower()

    if "pack" in format_str or has_packed:
        return "pack-quantized"
    if "int" in format_str or (has_scale and not has_packed):
        return "int-quantized"
    return "unquantized"

def repair_missing_weight_shape(index_data):
    repaired = {
        "metadata": dict(index_data.get("metadata", {})),
        "weight_map": dict(index_data.get("weight_map", {})),
        "tensor_metadata": {}
    }
    raw_tm = index_data.get("tensor_metadata", {})

    for name, meta in raw_tm.items():
        entry = dict(meta)
        if "weight_shape" not in entry and name.endswith(".weight"):
            bits = entry.get("bits", 8)
            packed_bytes = entry.get("packed_byte_size", 0)
            scale_shape = entry.get("scale_shape", [])

            if scale_shape and packed_bytes > 0:
                pack_factor = 8 // bits if bits < 8 else 1
                total_elements = (packed_bytes * pack_factor)
                if len(scale_shape) >= 2:
                    out_dim = scale_shape[0]
                    in_dim = total_elements // out_dim
                    entry["weight_shape"] = [out_dim, in_dim]
                elif len(scale_shape) == 1:
                    entry["weight_shape"] = [scale_shape[0], total_elements // scale_shape[0]]
                else:
                    entry["weight_shape"] = [total_elements]
        repaired["tensor_metadata"][name] = entry

    return repaired

def calculate_quant_byte_size(shape, num_bits, packed):
    total_elements = 1
    for d in shape:
        total_elements *= d

    if packed:
        elements_per_byte = 8 // num_bits
        return math.ceil(total_elements / elements_per_byte)
    else:
        bytes_per_element = math.ceil(num_bits / 8)
        return total_elements * bytes_per_element
