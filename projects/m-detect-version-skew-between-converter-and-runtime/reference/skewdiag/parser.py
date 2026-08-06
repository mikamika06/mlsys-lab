"""GGUF header and metadata model representation."""

GGUF_MAGIC = 0x46554747

def parse_model_manifest(data: dict) -> dict:
    magic = data.get("magic", 0)
    version = data.get("version", 0)
    meta = data.get("metadata", {})
    tensors = data.get("tensors", [])

    tensor_types = sorted(list({t.get("type") for t in tensors if "type" in t}))
    tensor_names = sorted(list({t.get("name") for t in tensors if "name" in t}))

    return {
        "valid_magic": magic == GGUF_MAGIC,
        "container_version": version,
        "architecture": meta.get("general.architecture", "unknown"),
        "converter_version": meta.get("general.version", "0.0.0"),
        "alignment": meta.get("general.alignment", 32),
        "tensor_types": tensor_types,
        "tensor_names": tensor_names,
        "metadata_keys": sorted(list(meta.keys()))
    }
