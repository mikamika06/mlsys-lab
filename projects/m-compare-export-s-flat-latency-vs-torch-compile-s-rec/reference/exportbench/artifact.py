import json
import struct


MAGIC_HEADER = b"PTEX"


def serialize_export_artifact(graph_data, weights):
    graph_bytes = json.dumps(graph_data).encode("utf-8")
    weights_bytes = weights.tobytes()

    header = struct.pack("<4sII", MAGIC_HEADER, len(graph_bytes), len(weights_bytes))
    return header + graph_bytes + weights_bytes


def deserialize_export_artifact(payload_bytes):
    if len(payload_bytes) < 12:
        raise ValueError("Corrupted artifact: Header missing or truncated.")

    magic, graph_len, weights_len = struct.unpack("<4sII", payload_bytes[:12])
    if magic != MAGIC_HEADER:
        raise ValueError(f"Invalid magic header: {magic}")

    expected_total = 12 + graph_len + weights_len
    if len(payload_bytes) < expected_total:
        raise ValueError(f"Truncated payload: Expected {expected_total} bytes, got {len(payload_bytes)}")

    graph_raw = payload_bytes[12 : 12 + graph_len]
    weights_raw = payload_bytes[12 + graph_len : expected_total]

    try:
        graph_data = json.loads(graph_raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse graph metadata: {e}")

    return graph_data, weights_raw
