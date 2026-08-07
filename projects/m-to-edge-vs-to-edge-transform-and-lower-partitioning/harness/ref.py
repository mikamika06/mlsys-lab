import struct

CONFIGS = [
    {"delegation_ratio": 0.5, "nodes": ["op1", "op2", "op3", "op4"]},
    {"delegation_ratio": 0.25, "nodes": ["op1", "op2", "op3", "op4"]},
    {"delegation_ratio": 1.0, "nodes": ["op1", "op2"]},
]

def partition_graph(graph_module, config):
    ratio = config.get("delegation_ratio", 0.0)
    nodes = graph_module.get("nodes", [])
    delegated_count = int(len(nodes) * ratio)
    delegated = nodes[:delegated_count]
    host = nodes[delegated_count:]
    return {"delegated": delegated, "host": host, "ratio": ratio}

def build_pte(payload):
    magic = b"\x50\x54\x45\x31"
    encoded_payload = payload.encode("utf-8") if isinstance(payload, str) else payload
    header = struct.pack("<I", len(encoded_payload))
    return magic + header + encoded_payload
