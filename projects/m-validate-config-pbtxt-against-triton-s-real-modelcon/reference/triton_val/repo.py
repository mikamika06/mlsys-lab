from typing import Dict, Any


def classify_repo_failure(repo: Dict[str, Any], model_name: str) -> str:
    if model_name not in repo:
        return "MISSING_MODEL_DIR"

    model_dir = repo[model_name]
    if not isinstance(model_dir, dict):
        return "INVALID_MODEL_DIR"

    if "config.pbtxt" not in model_dir:
        return "MISSING_CONFIG"

    config = model_dir["config.pbtxt"]
    if config is None or config == "MALFORMED":
        return "INVALID_CONFIG"

    version_dirs = [k for k, v in model_dir.items() if k.isdigit() and isinstance(v, dict)]
    if not version_dirs:
        return "MISSING_VERSION_DIR"

    has_model_file = False
    for v in version_dirs:
        v_files = model_dir[v]
        if any(f.startswith("model.") for f in v_files.keys()):
            has_model_file = True
            break

    if not has_model_file:
        return "MISSING_MODEL_FILE"

    return "OK"
