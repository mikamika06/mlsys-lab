import re


def parse_log(log_text):
    errors = []
    for line in log_text.splitlines():
        if "I[Model repository failed]" in line or "E[" in line or "error" in line.lower():
            match = re.search(r"model\s+['\"]?([a-zA-Z0-9_-]+)['\"]?", line)
            model_name = match.group(1) if match else "unknown"
            errors.append({"model": model_name, "message": line.strip()})
    return errors
