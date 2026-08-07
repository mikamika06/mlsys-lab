def parse_profiler_table(rows):
    res = {}
    for r in rows:
        name = r["Name"]
        val_str = r["Self CPU total"]
        if val_str.endswith("ms"):
            val = float(val_str[:-2])
        elif val_str.endswith("us"):
            val = float(val_str[:-2]) / 1000.0
        else:
            val = float(val_str)
        res[name] = val
    return res
