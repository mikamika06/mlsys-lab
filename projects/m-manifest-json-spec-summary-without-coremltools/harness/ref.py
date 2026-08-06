import os
import tempfile
import json
import struct
from mlpackage.manifest import summarize_manifest as ref_summarize
from mlpackage.bytes import attribute_bytes as ref_attribute
from mlpackage.mil import parse_mil_header as ref_parse_mil


def create_mock_package():
    tmp = tempfile.TemporaryDirectory()
    manifest = {
        "fileRootEntries": [
            {"key": "rootModel", "value": {"path": "Data/model.mlmodel"}},
            {"key": "weights", "value": {"path": "Data/weights.bin"}}
        ]
    }
    os.makedirs(os.path.join(tmp.name, "Data"), exist_ok=True)
    with open(os.path.join(tmp.name, "Manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(tmp.name, "Data", "model.mlmodel"), "wb") as f:
        f.write(b"modeldata" * 10)
    with open(os.path.join(tmp.name, "Data", "weights.bin"), "wb") as f:
        header = struct.pack("<4sIII", b"MIL1", 1, 32, 0)
        f.write(header + b"\x01" * 128)
    return tmp
