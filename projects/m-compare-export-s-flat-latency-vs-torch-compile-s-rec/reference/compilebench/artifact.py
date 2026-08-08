import json
import struct
import zlib


class ArtifactError(Exception):
    """Base exception for export artifact issues."""


class InvalidMagicError(ArtifactError):
    """Raised when artifact magic bytes do not match expected header."""


class TruncatedArtifactError(ArtifactError):
    """Raised when artifact data ends prematurely."""


class CorruptedArtifactError(ArtifactError):
    """Raised when artifact checksum or offset integrity check fails."""


MAGIC = b"PTEX"
HEADER_SIZE = 16


def serialize_export_artifact(graph_spec):
    """Serialize graph specification into binary export format."""
    payload = json.dumps(graph_spec).encode("utf-8")
    payload_len = len(payload)
    table_offset = HEADER_SIZE
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    header = struct.pack(">4sHHII", MAGIC, 1, 0, table_offset, payload_len)
    footer = struct.pack(">I", crc)
    return header + payload + footer


def deserialize_export_artifact(data):
    """Validate and deserialize binary export artifact."""
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
