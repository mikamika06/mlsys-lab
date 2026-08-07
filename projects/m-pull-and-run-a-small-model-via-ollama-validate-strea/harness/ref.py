import json
import hashlib

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "value": {"type": "integer"}
    },
    "required": ["status", "value"]
}

def validate_stream_chunk(chunk_str, schema):
    try:
        data = json.loads(chunk_str)
    except json.JSONDecodeError:
        return False
    for req in schema.get("required", []):
        if req not in data:
            return False
    props = schema.get("properties", {})
    for k, v in data.items():
        if k in props:
            ptype = props[k].get("type")
            if ptype == "string" and not isinstance(v, str):
                return False
            if ptype == "integer" and not isinstance(v, int):
                return False
    return True

def compute_blob_hash(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()

def compare_performance(ollama_tps, llama_cpp_tps):
    return {
        "ollama_tps": float(ollama_tps),
        "llama_cpp_tps": float(llama_cpp_tps),
        "ratio": float(ollama_tps / (llama_cpp_tps + 1e-9))
    }
