def analyze_log_truncation(lines):
    slot = 0
    shifted_tokens = 0
    kept_tokens = 0
    truncated = False
    for line in lines:
        if "slot " in line and "shift context" in line:
            parts = line.split()
            for idx, p in enumerate(parts):
                if p == "slot":
                    slot = int(parts[idx+1].rstrip(":"))
                if p == "shifted":
                    shifted_tokens = int(parts[idx+1])
                if p == "kept":
                    kept_tokens = int(parts[idx+1])
        if "context truncated" in line:
            truncated = True
            parts = line.split()
            for idx, p in enumerate(parts):
                if p == "slot":
                    slot = int(parts[idx+1].rstrip("."))
    return {"slot": slot, "shifted_tokens": shifted_tokens, "kept_tokens": kept_tokens, "truncated": truncated}
