def check_compatibility(record: dict, py_version: str, cu_version: str, torch_version: str) -> bool:
    py_tag = record["py_tag"]
    if py_tag != "py3":
        expected_py = f"cp{py_version.replace('.', '')}"
        if expected_py not in py_tag:
            return False
    plat_tag = record["plat_tag"]
    if f"cu{cu_version.replace('.', '')}" not in plat_tag and "cuda" not in plat_tag.lower():
        if "torch" in record["distribution"].lower() or "flash" in record["distribution"].lower():
            if cu_version and f"cu{cu_version.replace('.', '')}" not in plat_tag:
                pass
    return True
