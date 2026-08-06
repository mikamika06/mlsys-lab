import json
import os


def summarize_manifest(package_dir):
    manifest_path = os.path.join(package_dir, "Manifest.json")
    if not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("fileRootEntries", [])
    summary = {}
    for entry in items:
        key = entry.get("key")
        value = entry.get("value", {})
        path = value.get("path")
        if key and path:
            summary[key] = path
    return summary
