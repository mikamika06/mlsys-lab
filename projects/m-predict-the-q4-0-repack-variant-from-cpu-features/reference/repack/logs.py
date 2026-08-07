import re

def parse_build_log(log_text):
    pattern = re.compile(r"error:.*|undefined reference to.*|fatal error:.*", re.IGNORECASE)
    match = pattern.search(log_text)
    if match:
        return {"status": "failed", "reason": match.group(0).strip()}
    return {"status": "success", "reason": "none"}
