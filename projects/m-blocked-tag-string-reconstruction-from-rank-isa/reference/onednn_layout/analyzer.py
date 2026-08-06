from .parser import parse_verbose_log


def count_reorders(log_content: str) -> dict:
    events = parse_verbose_log(log_content)
    nchw_count = 0
    nhwc_count = 0
    for ev in events:
        src = ev["src_format"].lower()
        dst = ev["dst_format"].lower()
        if "nchw" in src or "nchw" in dst:
            nchw_count += 1
        if "nhwc" in src or "nhwc" in dst:
            nhwc_count += 1
    return {"nchw_reorders": nchw_count, "nhwc_reorders": nhwc_count}
