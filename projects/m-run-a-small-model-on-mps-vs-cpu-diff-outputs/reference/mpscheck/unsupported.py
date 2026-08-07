"""Extract unsupported MPS operations."""
import re

def extract_unsupported_op(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return None
    except Exception as e:
        msg = str(e)
        match = re.search(r"The operator '([^']+)' is not implemented for the MPS device", msg)
        if match:
            return match.group(1)
        if "NotImplementedError" in type(e).__name__ or "MPS" in msg:
            parts = msg.split("'")
            if len(parts) >= 2:
                return parts[1]
        raise e
