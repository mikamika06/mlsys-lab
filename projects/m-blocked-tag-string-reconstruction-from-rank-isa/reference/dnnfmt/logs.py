def count_reorders(log_text: str) -> dict:
    nchw_count = 0
    nhwc_count = 0
    for line in log_text.splitlines():
        if "reorder" in line.lower():
            if "nchw" in line.lower():
                nchw_count += 1
            if "nhwc" in line.lower():
                nhwc_count += 1
    return {"nchw": nchw_count, "nhwc": nhwc_count}
