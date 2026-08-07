import os


def reconstruct_repo(model_name, files, config_content):
    actions = []
    has_version = any(os.path.basename(f).isdigit() for f in files)
    if not has_version:
        actions.append({"action": "create_version_dir", "path": f"{model_name}/1"})

    has_config = any("config.pbtxt" in f for f in files)
    if not has_config:
        actions.append({"action": "write_config", "content": config_content})

    return actions
