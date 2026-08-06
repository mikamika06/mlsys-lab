import json
import gzip

def compute_save_sizes(model_config):
    weights = model_config.get("weights", {})
    frozen_bytes = 0
    for name, shape in weights.items():
        nelem = 1
        for dim in shape:
            nelem *= dim
        frozen_bytes += nelem * 2
    raw_payload = json.dumps(model_config).encode("utf-8")
    compressed_bytes = len(gzip.compress(raw_payload))
    ratio = float(compressed_bytes) / float(max(1, frozen_bytes))
    return {
        "frozen_size": frozen_bytes,
        "compressed_size": compressed_bytes,
        "size_ratio": ratio
    }
