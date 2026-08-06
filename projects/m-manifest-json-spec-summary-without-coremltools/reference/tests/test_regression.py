import os
import tempfile
import json
import struct
from mlpackage.manifest import summarize_manifest
from mlpackage.bytes import attribute_bytes
from mlpackage.mil import parse_mil_header


def test_manifest_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        manifest_data = {
            "fileRootEntries": [
                {"key": "model", "value": {"path": "Data/model.mlmodel"}}
            ]
        }
        with open(os.path.join(tmp, "Manifest.json"), "w") as f:
            json.dump(manifest_data, f)
        res = summarize_manifest(tmp)
        assert res.get("model") == "Data/model.mlmodel"


def test_byte_attribution():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "Data")
        os.makedirs(data_dir)
        file_path = os.path.join(data_dir, "weights.bin")
        with open(file_path, "wb") as f:
            f.write(b"\x00" * 100)
        res = attribute_bytes(tmp)
        assert res.get("Data/weights.bin") == 100
        assert res.get("_total") >= 100


def test_mil_header():
    header_bytes = struct.pack("<4sIII", b"MIL1", 2, 64, 1)
    res = parse_mil_header(header_bytes + b"\x00" * 64)
    assert res["magic"] == "MIL1"
    assert res["version"] == 2
    assert res["payload_offset"] == 64
