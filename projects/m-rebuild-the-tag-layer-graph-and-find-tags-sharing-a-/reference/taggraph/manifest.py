import json


def parse_manifest(raw_json):
    data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    layers = []
    for layer in data.get("layers", []):
        layers.append({
            "mediaType": layer.get("mediaType"),
            "digest": layer.get("digest"),
            "size": int(layer.get("size", 0))
        })
    return {
        "schemaVersion": data.get("schemaVersion"),
        "mediaType": data.get("mediaType"),
        "config": data.get("config"),
        "layers": layers
    }
