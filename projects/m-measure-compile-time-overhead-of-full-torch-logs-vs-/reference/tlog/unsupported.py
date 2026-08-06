def extract_unsupported_op(log_text: str) -> str:
    for line in log_text.splitlines():
        if "unsupported" in line or "failed" in line:
            parts = line.split("aten.")
            if len(parts) > 1:
                return "aten." + parts[1].split()[0].strip(".,;")
    for line in log_text.splitlines():
        if "aten." in line:
            parts = line.split("aten.")
            if len(parts) > 1:
                return "aten." + parts[1].split()[0].strip(".,;")
    return ""
