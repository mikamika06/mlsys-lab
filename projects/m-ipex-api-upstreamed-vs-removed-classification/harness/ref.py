"""Reference data and reference implementations for harness checks."""

API_TEST_CASES = [
    "ipex.optimize",
    "ipex.cpu.optimize",
    "ipex.quantization.prepare",
    "ipex.quantization.convert",
    "ipex.llm.optimize",
    "ipex.reflection",
    "ipex.core.enable_auto_dnnl",
    "ipex.core.disable_auto_dnnl",
    "ipex.enable_onednn_fusion",
    "ipex.disable_onednn_fusion",
    "ipex.align_dense_memory",
    "ipex.fast_bert",
    "ipex.nn.functional.interaction",
    "ipex.optimize_transformers",
]

UPSTREAMED_MAP = {
    "ipex.optimize": "torch.xpu.optimize",
    "ipex.cpu.optimize": "torch.compile",
    "ipex.quantization.prepare": "torch.ao.quantization.prepare",
    "ipex.quantization.convert": "torch.ao.quantization.convert",
    "ipex.llm.optimize": "torch.compile",
    "ipex.reflection": "torch.jit.trace",
}

REMOVED_SET = {
    "ipex.core.enable_auto_dnnl",
    "ipex.core.disable_auto_dnnl",
    "ipex.enable_onednn_fusion",
    "ipex.disable_onednn_fusion",
    "ipex.align_dense_memory",
}

RETAINED_SET = {
    "ipex.fast_bert",
    "ipex.nn.functional.interaction",
    "ipex.optimize_transformers",
}


def classify_api_call(api_name):
    if api_name in UPSTREAMED_MAP:
        return {"status": "upstreamed", "target": UPSTREAMED_MAP[api_name]}
    if api_name in REMOVED_SET:
        return {"status": "removed", "target": None}
    if api_name in RETAINED_SET:
        return {"status": "retained", "target": api_name}
    return {"status": "unknown", "target": None}


def classify_api_batch(api_list):
    return [classify_api_call(name) for name in api_list]


MANUAL_GRAPH = {
    "nodes": [
        {"id": 0, "op": "input", "output_bytes": 602112},
        {"id": 1, "op": "to", "target_format": "channels_last", "output_bytes": 602112},
        {"id": 2, "op": "conv2d", "weight_bytes": 36864, "output_bytes": 2408448},
        {"id": 3, "op": "relu", "output_bytes": 2408448},
        {"id": 4, "op": "to", "target_format": "channels_last", "output_bytes": 2408448},
        {"id": 5, "op": "conv2d", "weight_bytes": 147456, "output_bytes": 2408448},
    ]
}

IPEX_GRAPH = {
    "nodes": [
        {"id": 0, "op": "input", "output_bytes": 602112},
        {"id": 1, "op": "ipex_conv2d_relu", "weight_bytes": 36864, "output_bytes": 2408448},
        {"id": 2, "op": "ipex_conv2d", "weight_bytes": 147456, "output_bytes": 2408448},
    ]
}


def analyze_layout_conversions(graph):
    nodes = graph.get("nodes", [])
    count = 0
    for node in nodes:
        op = node.get("op", "")
        if op in ("to", "to_memory_format", "contiguous"):
            target_fmt = node.get("target_format") or node.get("kwargs", {}).get("memory_format")
            if target_fmt == "channels_last" or "channels_last" in str(target_fmt):
                count += 1
    return count


def diff_op_graphs(manual_graph, ipex_graph):
    manual_nodes = manual_graph.get("nodes", [])
    ipex_nodes = ipex_graph.get("nodes", [])

    manual_copies = analyze_layout_conversions(manual_graph)
    ipex_copies = analyze_layout_conversions(ipex_graph)
    redundant_copies_removed = max(0, manual_copies - ipex_copies)

    manual_weights_mem = sum(n.get("weight_bytes", 0) for n in manual_nodes)
    ipex_weights_mem = sum(n.get("weight_bytes", 0) for n in ipex_nodes)

    manual_copies_mem = sum(
        n.get("output_bytes", 0)
        for n in manual_nodes
        if n.get("op") in ("to", "to_memory_format", "contiguous")
    )
    ipex_copies_mem = sum(
        n.get("output_bytes", 0)
        for n in ipex_nodes
        if n.get("op") in ("to", "to_memory_format", "contiguous")
    )

    mem_saved = (manual_weights_mem + manual_copies_mem) - (ipex_weights_mem + ipex_copies_mem)

    return {
        "manual_node_count": len(manual_nodes),
        "ipex_node_count": len(ipex_nodes),
        "redundant_copies_removed": redundant_copies_removed,
        "memory_saved_bytes": max(0, mem_saved),
        "is_ipex_optimized": ipex_copies == 0 and len(ipex_nodes) < len(manual_nodes),
    }
