import math

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
