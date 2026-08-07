import numpy as np

UNFOLD_TESTS = [
    {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 12), "kernel_size": 3, "stride": 1, "padding": 0, "dilation": 1},
    {"x": np.arange(48, dtype=np.float32).reshape(2, 2, 12), "kernel_size": 3, "stride": 2, "padding": 1, "dilation": 1},
    {"x": np.arange(30, dtype=np.float32).reshape(1, 1, 30), "kernel_size": 5, "stride": 2, "padding": 2, "dilation": 2},
    {"x": np.arange(60, dtype=np.float32).reshape(2, 3, 10), "kernel_size": 2, "stride": 1, "padding": 0, "dilation": 1},
    {"x": np.arange(18, dtype=np.float32).reshape(1, 1, 18), "kernel_size": 3, "stride": 3, "padding": 0, "dilation": 1}
]

def ref_rewrite_unfold(x, kernel_size, stride, padding, dilation):
    batch, channels, length = x.shape
    out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    padded_length = length + 2 * padding
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")

    patches = []
    for i in range(out_length):
        start = i * stride
        chunk = padded[:, :, start:start + dilation * kernel_size:dilation]
        patches.append(chunk)
    stacked = np.stack(patches, axis=-1)
    reshaped = stacked.reshape(batch, channels * kernel_size, out_length)
    return reshaped

TRACE_TESTS = [
    {
        "tb": "File \"/opt/framework/core.py\", line 102, in lower\n  raise RuntimeError(\"unsupported op\")\nFile \"/user/project/model.py\", line 45, in forward\n  out = torch.nn.functional.unfold(x, 3)\nFile \"/opt/framework/export.py\", line 12, in export",
        "user_prefix": "/user/project/",
        "want": {"file": "/user/project/model.py", "line": 45}
    },
    {
        "tb": "File \"/lib/sys.py\", line 10, in run\nFile \"/home/dev/app/net.py\", line 88, in compute\n  y = custom_op(x)\nFile \"/lib/sys.py\", line 99, in exit",
        "user_prefix": "/home/dev/app/",
        "want": {"file": "/home/dev/app/net.py", "line": 88}
    },
    {
        "tb": "File \"/app/src/main.py\", line 15, in main\n  model(x)\nFile \"/app/src/layers.py\", line 22, in layer\n  unfold(x)",
        "user_prefix": "/app/src/",
        "want": {"file": "/app/src/layers.py", "line": 22}
    },
    {
        "tb": "File \"/code/runner.py\", line 5, in <module>\n  run()\nFile \"/code/engine/transformer.py\", line 77, in block\n  attn(q, k, v)",
        "user_prefix": "/code/engine/",
        "want": {"file": "/code/engine/transformer.py", "line": 77}
    },
    {
        "tb": "File \"/pkg/ext.py\", line 1, in test\nFile \"/project/src/inference.py\", line 34, in predict\n  export_model()",
        "user_prefix": "/project/src/",
        "want": {"file": "/project/src/inference.py", "line": 34}
    }
]

def ref_minimal_source_trace(tb_text, user_prefix):
    lines = tb_text.split("\n")
    best = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if "File \"" in line and user_prefix in line:
            parts = line.split('"')
            filename = parts[1]
            line_part = line.split(",")[1].strip()
            lineno = int(line_part.split()[1])
            best = {"file": filename, "line": lineno}
        i += 1
    return best

ESCAPE_FAILURES = [
    "UNSUPPORTED_OP_UNFOLD",
    "DYNAMIC_SHAPE_MISMATCH",
    "QUANTIZATION_SCALE_OVERFLOW",
    "CUSTOM_KERNEL_NOT_FOUND",
    "ATTENTION_MASK_RANK_MISMATCH",
    "RMSNORM_AXIS_OUT_OF_BOUNDS",
    "KV_CACHE_STRIDE_INVALID",
    "SILU_FUSION_UNSUPPORTED",
    "EMBEDDING_TABLE_TOO_LARGE",
    "ROPE_FREQ_BASE_INVALID"
]

def ref_choose_escape_hatch(error_msg):
    mapping = {
        "UNSUPPORTED_OP_UNFOLD": "rewrite_reshape_gather",
        "DYNAMIC_SHAPE_MISMATCH": "static_shape_pad",
        "QUANTIZATION_SCALE_OVERFLOW": "recompute_scale",
        "CUSTOM_KERNEL_NOT_FOUND": "fallback_aten_op",
        "ATTENTION_MASK_RANK_MISMATCH": "broadcast_mask",
        "RMSNORM_AXIS_OUT_OF_BOUNDS": "normalize_axis",
        "KV_CACHE_STRIDE_INVALID": "contiguous_cache",
        "SILU_FUSION_UNSUPPORTED": "split_silu_mul",
        "EMBEDDING_TABLE_TOO_LARGE": "shard_embedding",
        "ROPE_FREQ_BASE_INVALID": "default_rope_base"
    }
    return mapping.get(error_msg, "generic_fallback")
