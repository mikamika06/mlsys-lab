def classify_transcript(transcript_text):
    text = transcript_text.lower()
    if "err_nvgpuctrperm" in text or "profiling features are restricted" in text or "rmprofilingadminonly" in text:
        return "ERR_NVGPUCTRPERM"
    if "cuda_error_insufficient_driver" in text or "driver version is insufficient" in text or "mismatch between driver and library" in text:
        return "DRIVER_VERSION_MISMATCH"
    if "err_nvgpu_clock_throttling" in text or "hw slow down" in text or "sw thermal" in text or "thermal throttling" in text:
        return "THERMAL_THROTTLING"
    if "cuda_error_out_of_memory" in text or "alloc failed" in text or "out of memory" in text:
        return "OUT_OF_MEMORY"
    return "UNKNOWN_FAILURE"


def analyze_logs(transcripts_dict):
    results = {}
    for tid, text in transcripts_dict.items():
        results[tid] = classify_transcript(text)
    return results
