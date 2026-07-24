def _oracle(offload_optimizer, offload_param, num_params, param_bytes, optimizer_bytes_per_param):
    param_total = num_params * param_bytes
    optimizer_total = num_params * optimizer_bytes_per_param

    gpu = 0
    cpu = 0
    nvme = 0

    if offload_param == "nvme":
        nvme += param_total
    else:
        gpu += param_total

    if offload_optimizer == "cpu":
        cpu += optimizer_total
    elif offload_optimizer == "nvme":
        nvme += optimizer_total
    else:
        gpu += optimizer_total

    return {
        "gpu": gpu,
        "cpu": cpu,
        "nvme": nvme,
    }


def grade(sol, fx) -> dict:
    cases = [
        ("none", "none", 1000, 2, 12),
        ("cpu", "none", 4096, 2, 16),
        ("nvme", "none", 12345, 4, 24),
        ("none", "nvme", 500, 2, 12),
        ("nvme", "nvme", 8192, 2, 14),
        ("cpu", "nvme", 777, 8, 32),
    ]

    ok = 1.0
    for case in cases:
        expected = _oracle(*case)
        try:
            got = sol.zero_residency(*case)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
