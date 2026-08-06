def classify_transcript(transcript: dict) -> dict:
    log = transcript.get("log", "")
    ps = transcript.get("ps", [])
    mem = transcript.get("memory", {})

    if "CUDA out of memory" in log or "OutOfMemoryError: CUDA" in log or "torch.cuda.OutOfMemoryError" in log:
        return {
            "failure_class": "CUDA_OOM",
            "root_cause": "Requested GPU memory allocation exceeds available VRAM.",
            "one_line_fix": "Reduce context length or enable quantization/tensor parallelism."
        }
    if "Out of memory: Kill process" in log or "Killed" in log or mem.get("free_ram_mb", 1000) < 100:
        return {
            "failure_class": "HOST_OOM",
            "root_cause": "System RAM exhausted causing process termination by OS OOM killer.",
            "one_line_fix": "Increase swap space or offload fewer layers to CPU host memory."
        }
    if "Address already in use" in log or any("8000" in p or "8080" in p or "11434" in p for p in ps if "LISTEN" in p):
        return {
            "failure_class": "PORT_COLLISION",
            "root_cause": "Target serving port is already bound by an existing process.",
            "one_line_fix": "Terminate conflicting process or specify a non-default host port."
        }
    if "SHA256 mismatch" in log or "corrupted frame" in log or "invalid header" in log:
        return {
            "failure_class": "WEIGHT_CORRUPTION",
            "root_cause": "Model weight checkpoint file is incomplete or corrupted on disk.",
            "one_line_fix": "Re-download the model snapshot using force redownload flags."
        }
    if "maximum context length" in log or "exceeds model context limit" in log:
        return {
            "failure_class": "CONTEXT_EXCEEDED",
            "root_cause": "Input prompt token count exceeds positional embedding context window.",
            "one_line_fix": "Truncate prompt sequence or increase sliding window attention limit."
        }
    if "CUDA driver version is insufficient" in log or "driver/library version mismatch" in log:
        return {
            "failure_class": "DRIVER_MISMATCH",
            "root_cause": "Installed NVIDIA CUDA driver is incompatible with PyTorch CUDA runtime.",
            "one_line_fix": "Upgrade host NVIDIA display drivers to match installed CUDA toolkit."
        }
    if "Permission denied" in log or "EACCES" in log:
        return {
            "failure_class": "PERMISSION_DENIED",
            "root_cause": "Runner process lacks read/write file access permissions to weight directory.",
            "one_line_fix": "Update directory ownership with chown or adjust execution permissions."
        }
    if "unsupported quantization type" in log or "GGML format version" in log:
        return {
            "failure_class": "UNSUPPORTED_QUANT",
            "root_cause": "Runner executable binary lacks kernel support for requested quantization format.",
            "one_line_fix": "Recompile runner binary with updated quantization backend support."
        }
    if "NCCL error: unhandled system error" in log or "watchdog timeout" in log:
        return {
            "failure_class": "NCCL_TIMEOUT",
            "root_cause": "Inter-GPU communication timed out over NCCL ring connection.",
            "one_line_fix": "Check P2P interconnects and set NCCL_IB_DISABLE=1 if using fallback."
        }
    if "No space left on device" in log or "/dev/shm full" in log:
        return {
            "failure_class": "SHARED_MEMORY_EXHAUSTED",
            "root_cause": "Shared memory segment /dev/shm filled up during tensor inter-process transfer.",
            "one_line_fix": "Mount /dev/shm with expanded size or pass --shm-size parameter to docker container."
        }

    return {
        "failure_class": "UNKNOWN",
        "root_cause": "Unspecified failure signature.",
        "one_line_fix": "Inspect detailed trace logs."
    }


def classify_all(transcripts: list[dict]) -> list[dict]:
    return [classify_transcript(t) for t in transcripts]
