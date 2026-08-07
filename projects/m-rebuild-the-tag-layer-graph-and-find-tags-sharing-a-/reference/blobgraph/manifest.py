import json

def parse_manifest(manifest_str):
    data = json.loads(manifest_str)
    return [
        {
            "mediaType": layer.get("mediaType"),
            "digest": layer.get("digest"),
            "size": layer.get("size")
        }
        for layer in data.get("layers", [])
    ]
