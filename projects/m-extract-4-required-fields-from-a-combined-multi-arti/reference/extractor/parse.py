REQUIRED_KEYS = ["graph_id", "node_count", "op_name", "compile_status"]

def extract_required_fields(log_text):
    results = []
    for line in log_text.splitlines():
        if "[TORCH_LOGS]" in line:
            parts = line.replace("[TORCH_LOGS]:", "").strip().split()
            item = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k in REQUIRED_KEYS:
                        item[k] = v
            if all(k in item for k in REQUIRED_KEYS):
                results.append(item)
    return results
