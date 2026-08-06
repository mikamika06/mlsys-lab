import os


def inspect_package(package_path):
    required_dirs = ["Data", "Weights"]
    required_files = ["Manifest.json"]

    dirs_present = all(os.path.isdir(os.path.join(package_path, d)) for d in required_dirs)
    files_present = all(os.path.isfile(os.path.join(package_path, f)) for f in required_files)

    manifest_path = os.path.join(package_path, "Manifest.json")
    manifest_valid = False
    if os.path.isfile(manifest_path):
        manifest_valid = os.path.getsize(manifest_path) > 0

    return {
        "structure_valid": bool(dirs_present and files_present and manifest_valid),
        "missing_components": [d for d in required_dirs if not os.path.isdir(os.path.join(package_path, d))]
    }
