import json
import os
import hashlib


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: str, root_dir: str) -> bool:
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception:
        return False

    files = manifest.get("files", {})
    if not files:
        return False

    for rel_path, expected_hash in files.items():
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(full_path):
            return False
        if compute_sha256(full_path) != expected_hash:
            return False
    return True
