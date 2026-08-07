import numpy as np


def estimate_ir_size(graph_spec: list[dict], precision: str) -> dict[str, int]:
    bytes_per_elem = 2 if precision == "FP16" else 4
    total_weights_bytes = 0
    xml_metadata_bytes = 128
    for node in graph_spec:
        xml_metadata_bytes += 64 + len(node.get("name", ""))
        weights = node.get("weights")
        if weights is not None:
            num_elements = int(np.prod(weights))
            total_weights_bytes += num_elements * bytes_per_elem
    return {
        "bin_bytes": total_weights_bytes,
        "xml_bytes": xml_metadata_bytes,
        "total_bytes": total_weights_bytes + xml_metadata_bytes,
    }


def calculate_fp16_fp32_ratio(graph_spec: list[dict]) -> float:
    fp32_info = estimate_ir_size(graph_spec, "FP32")
    fp16_info = estimate_ir_size(graph_spec, "FP16")
    if fp32_info["total_bytes"] == 0:
        return 0.0
    return float(fp16_info["total_bytes"] / fp32_info["total_bytes"])
