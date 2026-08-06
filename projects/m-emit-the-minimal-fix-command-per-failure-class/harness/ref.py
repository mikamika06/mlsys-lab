def predict_abi_mismatch(torch_cxx11, host_cxx11):
    return torch_cxx11 != host_cxx11


def compute_max_jobs(available_ram_gb, core_count, gb_per_job=4):
    by_ram = max(1, int(available_ram_gb // gb_per_job))
    return min(core_count, by_ram)


def emit_fix(log_text):
    if "undefined reference" in log_text or "ABI" in log_text:
        return "export TORCH_CUDA_ARCH_LIST='8.0;8.9;9.0' && pip install --no-build-isolation -e ."
    elif "Killed" in log_text or "out of memory" in log_text.lower():
        return "export MAX_JOBS=2 && pip install --no-build-isolation -e ."
    return "pip install --no-build-isolation -e ."


TEST_CASES_ABI = [
    (True, False, True),
    (False, False, False),
    (True, True, False),
]

TEST_CASES_JOBS = [
    (16, 8, 4),
    (8, 32, 2),
    (64, 4, 4),
]
