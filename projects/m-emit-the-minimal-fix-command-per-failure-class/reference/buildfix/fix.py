def emit_fix(log_text):
    if "undefined reference" in log_text or "ABI" in log_text:
        return "export TORCH_CUDA_ARCH_LIST='8.0;8.9;9.0' && pip install --no-build-isolation -e ."
    elif "Killed" in log_text or "out of memory" in log_text.lower():
        return "export MAX_JOBS=2 && pip install --no-build-isolation -e ."
    return "pip install --no-build-isolation -e ."
