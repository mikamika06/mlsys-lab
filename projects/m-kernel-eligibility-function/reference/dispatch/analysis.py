from dispatch.selector import dispatch_kernel, is_eligible


def label_checkpoints(kernels, checkpoints):
    """Labels the dispatched kernel name for each checkpoint."""
    labels = []
    for ckpt in checkpoints:
        labels.append(dispatch_kernel(kernels, ckpt))
    return labels


def resolve_minimal_remedies(kernels, checkpoints):
    """Returns a list of dicts describing the minimal config changes to restore fast path."""
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
