import re

COMPAT_CASES = [
    ("1.18.0", "12.4", "9.1.0"),
    ("1.18.0", "11.8", "8.6.0"),
    ("1.18.0", "11.0", "8.0.0"),
    ("1.17.0", "12.2", "8.8.0"),
    ("1.15.0", "12.0", "8.9.1"),
]

PREFLIGHT_CASES = [
    (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        {
            "ort_version": "1.18.0",
            "cuda_version": "12.4",
            "cudnn_version": "9.1",
            "device_count": 1,
        },
        True,
    ),
    (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
        {
            "ort_version": "1.18.0",
            "cuda_version": "12.4",
            "cudnn_version": "9.1",
            "device_count": 1,
        },
        True,
    ),
    (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
        {
            "ort_version": "1.18.0",
            "cuda_version": "12.4",
            "cudnn_version": "9.1",
            "device_count": 1,
        },
        False,
    ),
    (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        {
            "ort_version": "1.18.0",
            "cuda_version": "11.0",
            "cudnn_version": "8.0",
            "device_count": 1,
        },
        True,
    ),
    (["CPUExecutionProvider"], ["CPUExecutionProvider"], {}, True),
]

LOG_CASES = [
    """[V:onnxruntime:Default, session_state.cc:123] Node [Conv_0] placed on [CUDAExecutionProvider]
[V:onnxruntime:Default, session_state.cc:124] Node [Relu_1] placed on [CUDAExecutionProvider]
[V:onnxruntime:Default, session_state.cc:125] Node [Add_2] placed on [CUDAExecutionProvider]""",
    """[V:onnxruntime:Default, session_state.cc:123] Node [Conv_0] placed on [CUDAExecutionProvider]
[V:onnxruntime:Default, session_state.cc:124] Placed node Relu_1 on execution provider CPUExecutionProvider
[V:onnxruntime:Default, session_state.cc:125] Node Add_2 placed on provider CUDAExecutionProvider""",
    """[V:onnxruntime:Default, session_state.cc:200] Placed node MatMul_0 on provider CPUExecutionProvider
[V:onnxruntime:Default, session_state.cc:201] Placed node BiasAdd_1 on provider CPUExecutionProvider""",
    """[V:onnxruntime:Default, session_state.cc:300] Node [Reshape_0] placed on [TensorRTExecutionProvider]
[V:onnxruntime:Default, session_state.cc:301] Node [Conv_1] placed on [TensorRTExecutionProvider]
[V:onnxruntime:Default, session_state.cc:302] Node [CustomOp_2] placed on [CPUExecutionProvider]
[V:onnxruntime:Default, session_state.cc:303] Node [Softmax_3] placed on [TensorRTExecutionProvider]""",
]

LOG_DISTRIBUTION_CASES = [
    (LOG_CASES[0], "CUDAExecutionProvider"),
    (LOG_CASES[1], "CUDAExecutionProvider"),
    (LOG_CASES[2], "CUDAExecutionProvider"),
    (LOG_CASES[3], "TensorRTExecutionProvider"),
]


def parse_version(v_str):
    return tuple(int(x) for x in v_str.split("."))


def check_cuda_cudnn_compat(ort_version, cuda_version, cudnn_version):
    ort_tuple = parse_version(ort_version)
    cuda_tuple = parse_version(cuda_version)
    cudnn_tuple = parse_version(cudnn_version)

    ort_major_minor = ort_tuple[:2]

    supported_cuda = {
        (1, 15): [(11, 8), (12, 0)],
        (1, 16): [(11, 8), (12, 2)],
        (1, 17): [(11, 8), (12, 2)],
        (1, 18): [(11, 8), (12, 4)],
    }

    if ort_major_minor not in supported_cuda:
        return False, f"Unsupported ORT version {ort_version}"

    cuda_mm = cuda_tuple[:2]
    if cuda_mm not in supported_cuda[ort_major_minor]:
        return False, f"Unsupported CUDA {cuda_version} for ORT {ort_version}"

    if cuda_mm == (11, 8):
        if not ((8, 2) <= cudnn_tuple[:2] < (9, 0)):
            return False, f"Incompatible cuDNN {cudnn_version} for CUDA 11.8"
    elif cuda_mm == (12, 0):
        if not ((8, 8) <= cudnn_tuple[:2] < (9, 0)):
            return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.0"
    elif cuda_mm == (12, 2):
        if ort_major_minor == (1, 16):
            if not ((8, 8) <= cudnn_tuple[:2] < (9, 0)):
                return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.2"
        else:
            if not ((8, 9) <= cudnn_tuple[:2] < (9, 0)):
                return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.2"
    elif cuda_mm == (12, 4):
        if not ((9, 0) <= cudnn_tuple[:2] < (10, 0)):
            return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.4"

    return True, "Compatible"


def validate_preflight(
    requested_providers, available_providers, env_info, strict=True
):
    if not requested_providers:
        return {
            "status": "OK",
            "selected_ep": "CPUExecutionProvider",
            "reason": "Default CPU",
        }

    non_cpu = [p for p in requested_providers if p != "CPUExecutionProvider"]
    if not non_cpu:
        return {
            "status": "OK",
            "selected_ep": "CPUExecutionProvider",
            "reason": "CPU selected",
        }

    primary = non_cpu[0]

    if primary not in available_providers:
        if strict:
            return {
                "status": "FAILED",
                "selected_ep": None,
                "reason": f"Provider {primary} unavailable",
            }
        return {
            "status": "FALLBACK",
            "selected_ep": "CPUExecutionProvider",
            "reason": f"Provider {primary} unavailable",
        }

    if env_info.get("device_count", 0) <= 0:
        if strict:
            return {
                "status": "FAILED",
                "selected_ep": None,
                "reason": "No GPU device detected",
            }
        return {
            "status": "FALLBACK",
            "selected_ep": "CPUExecutionProvider",
            "reason": "No GPU device detected",
        }

    ok, reason = check_cuda_cudnn_compat(
        env_info.get("ort_version", "0.0.0"),
        env_info.get("cuda_version", "0.0"),
        env_info.get("cudnn_version", "0.0"),
    )

    if not ok:
        if strict:
            return {"status": "FAILED", "selected_ep": None, "reason": reason}
        return {
            "status": "FALLBACK",
            "selected_ep": "CPUExecutionProvider",
            "reason": reason,
        }

    return {
        "status": "OK",
        "selected_ep": primary,
        "reason": "Preflight check passed",
    }


PATTERN = re.compile(
    r"(?:Placed\s+node\s+\[?([^\s\]]+)\]?\s+on\s+(?:execution\s+provider|provider)?\s*\[?(\w+)\]?)"
    r"|(?:Node\s+\[?([^\s\]]+)\]?\s+placed\s+on\s+(?:execution\s+provider|provider)?\s*\[?(\w+)\]?)",
    re.IGNORECASE,
)


def parse_ep_node_counts(log_text):
    counts = {}
    for line in log_text.splitlines():
        match = PATTERN.search(line)
        if match:
            groups = match.groups()
            if groups[0] is not None:
                ep = groups[1]
            else:
                ep = groups[3]
            counts[ep] = counts.get(ep, 0) + 1
    return counts


def analyze_ep_distribution(log_text, target_ep):
    counts = parse_ep_node_counts(log_text)
    total_nodes = sum(counts.values())
    target_nodes = counts.get(target_ep, 0)
    fallback_nodes = total_nodes - target_nodes
    fallback_ratio = (
        float(fallback_nodes) / total_nodes if total_nodes > 0 else 0.0
    )
    is_pure = (fallback_nodes == 0) and (total_nodes > 0)
    return {
        "counts": counts,
        "total_nodes": total_nodes,
        "target_nodes": target_nodes,
        "fallback_nodes": fallback_nodes,
        "fallback_ratio": fallback_ratio,
        "is_pure": is_pure,
    }
