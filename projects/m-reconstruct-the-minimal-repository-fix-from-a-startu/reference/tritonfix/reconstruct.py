def reconstruct_fix(logs):
    fixes = []
    for line in logs:
        if "missing file" in line:
            parts = line.split("'")
            model = parts[1]
            file_name = line.split("missing file ")[1].strip("'")
            fixes.append({"model": model, "action": "create_file", "path": f"{model}/1/{file_name}"})
        elif "missing platform declaration" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "update_config", "path": f"{model}/config.pbtxt"})
        elif "directory is empty" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "create_dir", "path": f"{model}/2"})
        elif "missing config.pbtxt" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "create_file", "path": f"{model}/config.pbtxt"})
        elif "invalid version directory name" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "rename_dir", "path": f"{model}/1"})
    return sorted(fixes, key=lambda x: (x["model"], x["action"]))
