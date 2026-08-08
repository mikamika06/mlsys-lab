import json
import struct
import zlib

MAGIC = b"PTEX"
HEADER_SIZE = 16


class ArtifactError(Exception):
    """Base exception for export artifact issues."""


class InvalidMagicError(ArtifactError):
    """Raised when artifact magic bytes do not match expected header."""


class TruncatedArtifactError(ArtifactError):
    """Raised when artifact data ends prematurely."""


class CorruptedArtifactError(ArtifactError):
    """Raised when artifact checksum or offset integrity check fails."""


CONFIGS = [
    {
        "compile_cost_ms": 150.0,
        "base_exec_ms": 1.0,
        "per_batch_ms": 0.5,
        "export_overhead_ms": 0.05,
        "max_batch_size": 32,
    },
    {
        "compile_cost_ms": 300.0,
        "base_exec_ms": 2.5,
        "per_batch_ms": 1.2,
        "export_overhead_ms": 0.1,
        "max_batch_size": 64,
    },
    {
        "compile_cost_ms": 500.0,
        "base_exec_ms": 5.0,
        "per_batch_ms": 2.0,
        "export_overhead_ms": 0.2,
        "max_batch_size": 128,
    },
]

REQUEST_SEQUENCES = [
    [{"batch_size": 1}, {"batch_size": 4}, {"batch_size": 1}, {"batch_size": 8}, {"batch_size": 4}, {"batch_size": 1}],
    [{"batch_size": 16}, {"batch_size": 16}, {"batch_size": 16}, {"batch_size": 32}, {"batch_size": 16}],
    [{"batch_size": 2}, {"batch_size": 4}, {"batch_size": 8}, {"batch_size": 16}, {"batch_size": 32}, {"batch_size": 64}],
]


def compare_latency(requests, config):
    compile_cost = config["compile_cost_ms"]
    base_exec = config["base_exec_ms"]
    per_batch = config["per_batch_ms"]
    export_overhead = config["export_overhead_ms"]
    max_batch = config["max_batch_size"]

    seen_shapes = set()
    compile_latencies = []
    export_latencies = []
    recompile_count = 0

    for req in requests:
        b = req["batch_size"]
        if b > max_batch:
            raise ValueError(f"Batch size {b} exceeds max allowed {max_batch}")

        exec_time = base_exec + b * per_batch

        if b not in seen_shapes:
            compile_lat = exec_time + compile_cost
            seen_shapes.add(b)
            recompile_count += 1
        else:
            compile_lat = exec_time
        compile_latencies.append(compile_lat)

        export_lat = exec_time + export_overhead
        export_latencies.append(export_lat)

    max_compile = max(compile_latencies) if compile_latencies else 0.0
    max_export = max(export_latencies) if export_latencies else 1.0
    sum_compile = sum(compile_latencies)
    sum_export = sum(export_latencies) if export_latencies else 1.0

    return {
        "compile_latencies": compile_latencies,
        "export_latencies": export_latencies,
        "recompile_count": recompile_count,
        "max_spike_ratio": max_compile / max_export if max_export > 0 else 0.0,
        "total_latency_ratio": sum_compile / sum_export if sum_export > 0 else 0.0,
    }


def serialize_export_artifact(graph_spec):
    payload = json.dumps(graph_spec).encode("utf-8")
    payload_len = len(payload)
    table_offset = HEADER_SIZE
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    header = struct.pack(">4sHHII", MAGIC, 1, 0, table_offset, payload_len)
    footer = struct.pack(">I", crc)
    return header + payload + footer


def deserialize_export_artifact(data):
    if len(data) < HEADER_SIZE:
        raise TruncatedArtifactError("Data shorter than header size")

    magic, version, flags, table_offset, payload_len = struct.unpack(">4sHHII", data[:HEADER_SIZE])
    if magic != MAGIC:
        raise InvalidMagicError(f"Invalid magic: {magic}")

    expected_len = HEADER_SIZE + payload_len + 4
    if len(data) < expected_len:
        raise TruncatedArtifactError(f"Expected {expected_len} bytes, got {len(data)}")

    if table_offset != HEADER_SIZE:
        raise CorruptedArtifactError(f"Invalid table offset: {table_offset}")

    payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]
    expected_crc = struct.unpack(">I", data[HEADER_SIZE + payload_len:expected_len])[0]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        raise CorruptedArtifactError(f"CRC mismatch: expected {expected_crc}, got {actual_crc}")

    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as e:
        raise CorruptedArtifactError(f"Failed to decode payload: {e}") from e


GRAPH_SPECS = [
    {"nodes": ["in", "conv", "relu"], "batch_bounds": [1, 32]},
    {"nodes": ["in", "linear", "gelu", "out"], "batch_bounds": [1, 64]},
    {"nodes": ["in", "attn", "norm", "out"], "batch_bounds": [1, 128]},
]

VALID_ARTIFACTS = [serialize_export_artifact(s) for s in GRAPH_SPECS]


def build_corrupted_artifacts():
    out = []
    base = bytearray(VALID_ARTIFACTS[0])
    corrupted_crc = bytearray(base)
    corrupted_crc[HEADER_SIZE + 2] ^= 0xFF
    out.append(bytes(corrupted_crc))

    bad_offset = struct.pack(">4sHHII", MAGIC, 1, 0, 32, len(json.dumps(GRAPH_SPECS[0])))
    out.append(bad_offset + base[HEADER_SIZE:])
    return out


def build_truncated_artifacts():
    base = VALID_ARTIFACTS[0]
    return [base[:8], base[:HEADER_SIZE + 5]]


def build_invalid_magic_artifacts():
    base = bytearray(VALID_ARTIFACTS[0])
    base[0:4] = b"FAIL"
    return [bytes(base)]


CORRUPTED_ARTIFACTS = build_corrupted_artifacts()
TRUNCATED_ARTIFACTS = build_truncated_artifacts()
INVALID_MAGIC_ARTIFACTS = build_invalid_magic_artifacts()
