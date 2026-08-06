import random

KERNELS = [
    {
        "name": "fused_marlin_awq",
        "priority": 100,
        "allowed_in_dtypes": ["float16", "bfloat16"],
        "allowed_out_dtypes": ["float16", "bfloat16"],
        "quant_scheme": "awq_int4",
        "group_size": 128,
        "min_k": 128,
        "align_k": 64,
        "align_n": 64,
        "req_align_bytes": 16,
    },
    {
        "name": "cutlass_fp16_tensor_core",
        "priority": 80,
        "allowed_in_dtypes": ["float16"],
        "allowed_out_dtypes": ["float16"],
        "quant_scheme": "none",
        "group_size": None,
        "min_k": 64,
        "align_k": 16,
        "align_n": 16,
        "req_align_bytes": 16,
    },
    {
        "name": "gptq_int4_cuda",
        "priority": 70,
        "allowed_in_dtypes": ["float16"],
        "allowed_out_dtypes": ["float16"],
        "quant_scheme": "gptq_int4",
        "group_size": 128,
        "min_k": 128,
        "align_k": 32,
        "align_n": 32,
        "req_align_bytes": 8,
    },
]


def generate_checkpoints():
    rng = random.Random(42)
    checkpoints = []

    for i in range(20):
        if i % 4 == 0:
            ckpt = {
                "in_dtype": "float16",
                "out_dtype": "float16",
                "quant_scheme": "awq_int4",
                "group_size": 128,
                "k": 256,
                "n": 256,
                "ptr_align_bytes": 16,
            }
        elif i % 4 == 1:
            ckpt = {
                "in_dtype": "float16",
                "out_dtype": "float16",
                "quant_scheme": "none",
                "group_size": None,
                "k": 128,
                "n": 128,
                "ptr_align_bytes": 16,
            }
        elif i % 4 == 2:
            ckpt = {
                "in_dtype": "float16",
                "out_dtype": "float16",
                "quant_scheme": "awq_int4",
                "group_size": 64,
                "k": 256,
                "n": 256,
                "ptr_align_bytes": 16,
            }
        else:
            ckpt = {
                "in_dtype": "float16",
                "out_dtype": "float16",
                "quant_scheme": "none",
                "group_size": None,
                "k": 130,
                "n": 128,
                "ptr_align_bytes": 16,
            }
        checkpoints.append(ckpt)

    return checkpoints


CHECKPOINTS = generate_checkpoints()


def is_eligible(kernel_spec, layer_config):
    if layer_config.get("in_dtype") not in kernel_spec.get("allowed_in_dtypes", []):
        return False
    if layer_config.get("out_dtype") not in kernel_spec.get("allowed_out_dtypes", []):
        return False
    if layer_config.get("quant_scheme") != kernel_spec.get("quant_scheme"):
        return False

    req_group_size = kernel_spec.get("group_size")
    if req_group_size is not None and layer_config.get("group_size") != req_group_size:
        return False

    min_k = kernel_spec.get("min_k", 0)
    if layer_config.get("k", 0) < min_k:
        return False

    align_k = kernel_spec.get("align_k", 1)
    if layer_config.get("k", 0) % align_k != 0:
        return False

    align_n = kernel_spec.get("align_n", 1)
    if layer_config.get("n", 0) % align_n != 0:
        return False

    req_align_bytes = kernel_spec.get("req_align_bytes")
    if req_align_bytes is not None:
        ptr_align = layer_config.get("ptr_align_bytes", 1)
        if ptr_align % req_align_bytes != 0:
            return False

    return True


def dispatch_kernel(available_kernels, layer_config):
    eligible = [
        k for k in available_kernels
        if is_eligible(k, layer_config)
    ]
    if not eligible:
        return "fallback_gemm"
    eligible.sort(key=lambda x: x.get("priority", 0), reverse=True)
    return eligible[0]["name"]


def label_checkpoints(kernels, checkpoints):
    return [dispatch_kernel(kernels, ckpt) for ckpt in checkpoints]


def resolve_minimal_remedies(kernels, checkpoints):
    results = []
    target_kernels = [k for k in kernels if k["name"] != "fallback_gemm"]
    target_kernels.sort(key=lambda x: x.get("priority", 0), reverse=True)

    for ckpt in checkpoints:
        current_dispatched = dispatch_kernel(kernels, ckpt)
        if current_dispatched != "fallback_gemm":
            results.append({"needed_change": None, "target_kernel": current_dispatched})
            continue

        found_remedy = False
        for k_spec in target_kernels:
            diffs = {}
            if ckpt.get("in_dtype") not in k_spec.get("allowed_in_dtypes", []):
                diffs["in_dtype"] = k_spec["allowed_in_dtypes"][0]
            if ckpt.get("out_dtype") not in k_spec.get("allowed_out_dtypes", []):
                diffs["out_dtype"] = k_spec["allowed_out_dtypes"][0]
            if ckpt.get("quant_scheme") != k_spec.get("quant_scheme"):
                diffs["quant_scheme"] = k_spec["quant_scheme"]

            req_gs = k_spec.get("group_size")
            if req_gs is not None and ckpt.get("group_size") != req_gs:
                diffs["group_size"] = req_gs

            min_k = k_spec.get("min_k", 0)
            if ckpt.get("k", 0) < min_k:
                diffs["k"] = min_k

            align_k = k_spec.get("align_k", 1)
            if ckpt.get("k", 0) % align_k != 0:
                cur_k = ckpt.get("k", 0)
                diffs["k"] = ((cur_k + align_k - 1) // align_k) * align_k

            align_n = k_spec.get("align_n", 1)
            if ckpt.get("n", 0) % align_n != 0:
                cur_n = ckpt.get("n", 0)
                diffs["n"] = ((cur_n + align_n - 1) // align_n) * align_n

            req_align = k_spec.get("req_align_bytes")
            if req_align is not None and (ckpt.get("ptr_align_bytes", 1) % req_align != 0):
                diffs["ptr_align_bytes"] = req_align

            if len(diffs) == 1:
                key, new_val = list(diffs.items())[0]
                test_ckpt = dict(ckpt)
                test_ckpt[key] = new_val
                if is_eligible(k_spec, test_ckpt):
                    results.append({"needed_change": {key: new_val}, "target_kernel": k_spec["name"]})
                    found_remedy = True
                    break

        if not found_remedy:
            results.append({"needed_change": None, "target_kernel": "fallback_gemm"})

    return results
