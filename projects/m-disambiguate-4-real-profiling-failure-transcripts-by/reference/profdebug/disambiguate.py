def disambiguate(transcript):
    if "ERR_NVGPUCTRPERM" in transcript:
        return "nvgpuctrperm"
    if "version mismatch" in transcript:
        return "driver_mismatch"
    if "timed out" in transcript:
        return "timeout"
    return "out_of_memory"
