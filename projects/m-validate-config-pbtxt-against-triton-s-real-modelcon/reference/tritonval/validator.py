import re

def validate_config(config_text: str) -> dict:
    errors = []
    lines = config_text.splitlines()
    has_name = False
    has_platform_or_backend = False
    has_max_batch_size = False
    name_val = None
    platform_val = None
    backend_val = None
    max_batch_val = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("name:"):
            has_name = True
            m = re.search(r'name:\s*"([^"]+)"', stripped)
            if m:
                name_val = m.group(1)
        elif stripped.startswith("platform:"):
            has_platform_or_backend = True
            m = re.search(r'platform:\s*"([^"]+)"', stripped)
            if m:
                platform_val = m.group(1)
        elif stripped.startswith("backend:"):
            has_platform_or_backend = True
            m = re.search(r'backend:\s*"([^"]+)"', stripped)
            if m:
                backend_val = m.group(1)
        elif stripped.startswith("max_batch_size:"):
            has_max_batch_size = True
            m = re.search(r'max_batch_size:\s*(\d+)', stripped)
            if m:
                max_batch_val = int(m.group(1))

    if not has_name:
        errors.append("missing required field: name")
    elif not name_val:
        errors.append("invalid or empty name value")

    if not has_platform_or_backend:
        errors.append("missing required field: platform or backend")

    if not has_max_batch_size:
        errors.append("missing required field: max_batch_size")
    elif max_batch_val is not None and max_batch_val < 0:
        errors.append("max_batch_size cannot be negative")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }
