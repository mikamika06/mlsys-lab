from ortpreflight.oracle import check_cuda_cudnn_compat


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
