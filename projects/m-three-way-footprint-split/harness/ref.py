import math


def analyze_three_way_footprint(binary_info, runtime_config, model_tensors):
    text_size = binary_info.get("text_bytes", 0)
    rodata_size = binary_info.get("rodata_bytes", 0)
    data_size = binary_info.get("data_bytes", 0)
    bss_size = binary_info.get("bss_bytes", 0)

    static_binary = text_size + rodata_size + data_size + bss_size

    base_heap = runtime_config.get("base_heap_bytes", 0)
    ctx_structs = runtime_config.get("context_struct_bytes", 0)
    arena_overhead = runtime_config.get("arena_metadata_bytes", 0)

    runtime_infra = base_heap + ctx_structs + arena_overhead

    weights_bytes = 0
    activations_bytes = 0
    workspace_bytes = 0

    for tensor in model_tensors:
        numel = tensor.get("numel", 0)
        elem_bytes = tensor.get("elem_bytes", 1)
        total_bytes = numel * elem_bytes
        category = tensor.get("category", "activation")
        if category == "weight":
            weights_bytes += total_bytes
        elif category == "workspace":
            workspace_bytes += total_bytes
        else:
            activations_bytes += total_bytes

    dynamic_tensors = weights_bytes + activations_bytes + workspace_bytes
    total_footprint = static_binary + runtime_infra + dynamic_tensors

    return {
        "static_binary_bytes": static_binary,
        "runtime_infra_bytes": runtime_infra,
        "dynamic_tensors_bytes": dynamic_tensors,
        "total_footprint_bytes": total_footprint,
        "weight_bytes": weights_bytes,
        "activation_bytes": activations_bytes,
        "workspace_bytes": workspace_bytes
    }


def selective_registration_win(all_kernels, used_ops):
    used_set = set(used_ops)
    total_binary_before = 0
    total_binary_after = 0
    total_table_before = 0
    total_table_after = 0

    pruned_ops = []

    for k in all_kernels:
        op_name = k["op"]
        code_size = k.get("code_bytes", 0)
        table_size = k.get("table_bytes", 0)

        total_binary_before += code_size
        total_table_before += table_size

        if op_name in used_set:
            total_binary_after += code_size
            total_table_after += table_size
        else:
            pruned_ops.append(op_name)

    binary_saved = total_binary_before - total_binary_after
    table_saved = total_table_before - total_table_after
    total_saved = binary_saved + table_saved

    pct_binary_saved = (binary_saved / total_binary_before * 100.0) if total_binary_before > 0 else 0.0

    return {
        "binary_bytes_before": total_binary_before,
        "binary_bytes_after": total_binary_after,
        "binary_bytes_saved": binary_saved,
        "table_bytes_saved": table_saved,
        "total_bytes_saved": total_saved,
        "percent_binary_saved": pct_binary_saved,
        "pruned_ops": sorted(pruned_ops)
    }


def predict_peak_rss(execution_plan, alignment=64, overhead_bytes=0):
    if not execution_plan:
        return {
            "peak_rss_bytes": overhead_bytes,
            "peak_step": 0,
            "active_tensors_at_peak": []
        }

    max_step = 0
    for item in execution_plan:
        max_step = max(max_step, item.get("start_step", 0), item.get("end_step", 0))

    peak_rss = 0
    peak_step = 0
    peak_active = []

    for step in range(max_step + 1):
        current_rss = overhead_bytes
        current_active = []
        for item in execution_plan:
            start = item.get("start_step", 0)
            end = item.get("end_step", 0)
            if start <= step <= end:
                raw_size = item.get("size_bytes", 0)
                if alignment > 0:
                    aligned_size = math.ceil(raw_size / alignment) * alignment
                else:
                    aligned_size = raw_size
                current_rss += aligned_size
                current_active.append(item.get("name", "unnamed"))

        if current_rss > peak_rss:
            peak_rss = current_rss
            peak_step = step
            peak_active = sorted(current_active)

    return {
        "peak_rss_bytes": peak_rss,
        "peak_step": peak_step,
        "active_tensors_at_peak": peak_active
    }


SPLIT_TEST_CASES = [
    (
        {"text_bytes": 1000, "rodata_bytes": 200, "data_bytes": 50, "bss_bytes": 100},
        {"base_heap_bytes": 500, "context_struct_bytes": 80, "arena_metadata_bytes": 20},
        [
            {"numel": 1000, "elem_bytes": 4, "category": "weight"},
            {"numel": 200, "elem_bytes": 2, "category": "activation"},
            {"numel": 500, "elem_bytes": 4, "category": "workspace"}
        ]
    ),
    (
        {"text_bytes": 2048, "rodata_bytes": 512, "data_bytes": 128, "bss_bytes": 64},
        {"base_heap_bytes": 1024, "context_struct_bytes": 256, "arena_metadata_bytes": 64},
        [
            {"numel": 5000, "elem_bytes": 2, "category": "weight"},
            {"numel": 1000, "elem_bytes": 4, "category": "activation"}
        ]
    ),
    (
        {"text_bytes": 500, "rodata_bytes": 100, "data_bytes": 10, "bss_bytes": 20},
        {"base_heap_bytes": 200, "context_struct_bytes": 30, "arena_metadata_bytes": 10},
        []
    ),
    (
        {"text_bytes": 8192, "rodata_bytes": 1024, "data_bytes": 256, "bss_bytes": 512},
        {"base_heap_bytes": 4096, "context_struct_bytes": 512, "arena_metadata_bytes": 128},
        [
            {"numel": 10000, "elem_bytes": 4, "category": "weight"},
            {"numel": 2000, "elem_bytes": 4, "category": "activation"},
            {"numel": 1000, "elem_bytes": 4, "category": "workspace"}
        ]
    ),
    (
        {"text_bytes": 12000, "rodata_bytes": 3000, "data_bytes": 400, "bss_bytes": 800},
        {"base_heap_bytes": 8000, "context_struct_bytes": 1000, "arena_metadata_bytes": 500},
        [
            {"numel": 20000, "elem_bytes": 2, "category": "weight"}
        ]
    )
]

SELECTIVE_TEST_CASE = (
    [
        {"op": "conv2d", "code_bytes": 8192, "table_bytes": 256},
        {"op": "depthwise_conv2d", "code_bytes": 4096, "table_bytes": 128},
        {"op": "relu", "code_bytes": 512, "table_bytes": 32},
        {"op": "max_pool2d", "code_bytes": 1024, "table_bytes": 64},
        {"op": "softmax", "code_bytes": 2048, "table_bytes": 128},
        {"op": "add", "code_bytes": 256, "table_bytes": 16}
    ],
    ["conv2d", "relu", "add"]
)

PREDICTOR_TEST_CASE = (
    [
        {"name": "input", "start_step": 0, "end_step": 1, "size_bytes": 500},
        {"name": "conv1_w", "start_step": 0, "end_step": 5, "size_bytes": 2000},
        {"name": "conv1_out", "start_step": 1, "end_step": 3, "size_bytes": 1200},
        {"name": "relu1_out", "start_step": 2, "end_step": 4, "size_bytes": 1200},
        {"name": "fc1_w", "start_step": 4, "end_step": 5, "size_bytes": 1500},
        {"name": "output", "start_step": 5, "end_step": 5, "size_bytes": 100}
    ],
    64,
    1024
)
