from profops.parser import parse_profiler_table

def detect_sync_hotspot(rows):
    parsed = parse_profiler_table(rows)
    if "aten::copy_" in parsed and parsed["aten::copy_"] > 30.0:
        return "aten::copy_"
    return None
