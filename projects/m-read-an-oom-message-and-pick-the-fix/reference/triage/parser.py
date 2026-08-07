import re


def parse_oom_message(msg: str) -> dict:
    out = {}
    m_req = re.search(r"Tried to allocate ([\d\.]+)\s*([KMGT]iB)?", msg)
    if m_req:
        val = float(m_req.group(1))
        unit = m_req.group(2) or "B"
        mult = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
        out["requested_bytes"] = int(val * mult.get(unit, 1))

    m_cap = re.search(r"([\d\.]+)\s*([KMGT]iB)? total capacity", msg)
    if m_cap:
        val = float(m_cap.group(1))
        unit = m_cap.group(2) or "B"
        mult = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
        out["total_capacity_bytes"] = int(val * mult.get(unit, 1))

    m_alloc = re.search(r"([\d\.]+)\s*([KMGT]iB)? already allocated", msg)
    if m_alloc:
        val = float(m_alloc.group(1))
        unit = m_alloc.group(2) or "B"
        mult = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
        out["allocated_bytes"] = int(val * mult.get(unit, 1))

    m_res = re.search(r"([\d\.]+)\s*([KMGT]iB)? reserved in total", msg)
    if m_res:
        val = float(m_res.group(1))
        unit = m_res.group(2) or "B"
        mult = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
        out["reserved_bytes"] = int(val * mult.get(unit, 1))

    return out


def pick_fix(msg: str) -> str:
    info = parse_oom_message(msg)
    reserved = info.get("reserved_bytes", 0)
    allocated = info.get("allocated_bytes", 0)
    if reserved > 0 and (reserved - allocated) > (reserved * 0.25):
        if "max_split_size_mb" in msg.lower():
            return "set_max_split_size"
        return "empty_cache"
    return "reduce_batch_size"
