def generate_log_fixture():
    log = (
        "[00:01] INFO: Booting runner\n"
        "EVENT: bootstrap_env\n"
        "[00:02] INFO: Loading config\n"
        "EVENT: load_manifest\n"
        "[00:03] INFO: Allocating buffers\n"
        "EVENT: allocate_tensor_memory\n"
        "CRITICAL: Connection reset by peer during weight stream"
    )
    return log

def generate_error_contexts():
    return [
        {"message": "FileNotFoundError: weights/model.safetensors missing"},
        {"message": "OSError: model manifest not found in directory"},
    ]

def generate_metric_fixtures():
    return [
        {"device": "cuda", "cuda_kernels_executed": 0, "cpu_fallback_ops": 42},
        {"device": "cuda", "cuda_kernels_executed": 1000, "cpu_fallback_ops": 2},
    ]
