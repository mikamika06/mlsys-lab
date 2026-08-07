import os

def classify_layout(model_dir: str) -> str:
    if not os.path.exists(model_dir):
        return "missing_model_directory"

    config_path = os.path.join(model_dir, "config.pbtxt")
    if not os.path.exists(config_path):
        return "missing_config_pbtxt"

    entries = os.listdir(model_dir)
    version_dirs = []
    for entry in entries:
        if entry == "config.pbtxt":
            continue
        full_path = os.path.join(model_dir, entry)
        if os.path.isdir(full_path):
            if entry.isdigit():
                version_dirs.append(int(entry))
            else:
                return "malformed_version_directory_name"
        else:
            return "unexpected_file_in_model_directory"

    if not version_dirs:
        return "missing_version_directories"

    return "valid"
