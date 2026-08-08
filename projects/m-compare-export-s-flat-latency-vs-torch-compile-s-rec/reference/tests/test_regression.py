import sys
import pytest

sys.path.insert(0, ".")
from compilebench.artifact import (
    CorruptedArtifactError,
    InvalidMagicError,
    TruncatedArtifactError,
    deserialize_export_artifact,
    serialize_export_artifact,
)


def test_valid_artifact_roundtrip():
    spec = {"nodes": ["input", "linear", "relu"], "dynamic_shapes": {"batch": [1, 64]}}
    encoded = serialize_export_artifact(spec)
    decoded = deserialize_export_artifact(encoded)
    assert decoded == spec


def test_catches_invalid_magic():
    spec = {"nodes": ["input"]}
    encoded = bytearray(serialize_export_artifact(spec))
    encoded[0:4] = b"BADM"
    try:
        deserialize_export_artifact(bytes(encoded))
        assert False, "Should have raised InvalidMagicError"
    except InvalidMagicError:
        pass


def test_catches_truncation():
    spec = {"nodes": ["input"]}
    encoded = serialize_export_artifact(spec)
    try:
        deserialize_export_artifact(encoded[:10])
        assert False, "Should have raised TruncatedArtifactError"
    except TruncatedArtifactError:
        pass


def test_catches_payload_corruption():
    spec = {"nodes": ["input", "linear"]}
    encoded = bytearray(serialize_export_artifact(spec))
    encoded[18] ^= 0xFF
    try:
        deserialize_export_artifact(bytes(encoded))
        assert False, "Should have raised CorruptedArtifactError"
    except CorruptedArtifactError:
        pass
