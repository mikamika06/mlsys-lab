import re

FILENAME_REGEX = re.compile(r"^(.+)-000([1-9][0-9]*)-of-000([1-9][0-9]*)\.gguf$")


def validate_filename(filename: str) -> bool:
    match = FILENAME_REGEX.match(filename)
    if not match:
        return False
    part = int(match.group(2))
    total = int(match.group(3))
    return 1 <= part <= total


def merge_manifests(manifests: list) -> dict:
    if not manifests:
        return {}
    logical = {
        "version": manifests[0].get("version", 1),
        "tensors": {},
        "total_size": 0,
        "shard_count": len(manifests),
    }
    for idx, m in enumerate(manifests):
        for name, info in m.get("tensors", {}).items():
            if name in logical["tensors"]:
                raise ValueError(f"Duplicate tensor {name} found across shards")
            tensor_info = dict(info)
            tensor_info["shard_index"] = idx
            logical["tensors"][name] = tensor_info
        logical["total_size"] += m.get("size", 0)
    return logical
