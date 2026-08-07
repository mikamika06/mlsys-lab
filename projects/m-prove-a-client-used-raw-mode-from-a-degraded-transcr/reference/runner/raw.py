def prove_raw_mode(transcript):
    if not transcript:
        return False
    lines = transcript.splitlines()
    has_raw_flag = False
    has_raw_marker = False
    for line in lines:
        stripped = line.strip()
        if "X-Raw-Mode: true" in stripped or "mode=raw" in stripped:
            has_raw_flag = True
        if "[RAW_EXECUTION]" in stripped:
            has_raw_marker = True
    return has_raw_flag and has_raw_marker
