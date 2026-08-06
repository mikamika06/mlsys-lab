#!/usr/bin/env python3
"""Ground truth for the GGUF units, plus a deliberately damaged copy.

The learner parses slice.gguf with nothing but `struct`. What their parse is
compared against is recorded here by the gguf library itself, so the grade does
not rest on my reading of the format.

The damaged copy exists because validation is the part of this job that matters
in production: a checkpoint whose tensor index points past the end of the file
should be rejected in milliseconds, not discovered after a 40-second load. The
damage is one field, recorded exactly, so a checker can insist the learner names
the right one.

    python3 tools/record_gguf_truth.py
"""
import json
import os
import struct
import sys

import numpy as np

try:
    import gguf
    from gguf import GGUFReader, GGUFValueType
except ImportError:
    sys.exit("needs the `gguf` package")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "projects", "_fixtures", "gguf")


def jsonable(v, limit=2048):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        out = [jsonable(x) for x in list(v)[:limit]]
        return out
    return v


def main():
    path = os.path.join(FIX, "slice.gguf")
    if not os.path.isfile(path):
        sys.exit("run tools/record_gguf.py first")
    r = GGUFReader(path)

    kv = {}
    for key, field in r.fields.items():
        if key.startswith("GGUF."):
            continue
        if not field.types:
            continue
        head = field.types[0]
        try:
            val = field.contents()
        except Exception:
            continue
        kv[key] = {"type": head.name, "value": jsonable(val)}
        if head == GGUFValueType.ARRAY:
            kv[key]["length"] = len(val)
            kv[key]["element_type"] = (field.types[1].name if len(field.types) > 1
                                       else "STRING")

    tensors = []
    for t in r.tensors:
        qt = gguf.GGMLQuantizationType(int(t.tensor_type))
        tensors.append({
            "name": t.name,
            "ggml_type_id": int(t.tensor_type),
            "ggml_type": qt.name,
            "shape_ggml_order": [int(x) for x in t.shape.tolist()],
            "n_elements": int(np.prod([int(x) for x in t.shape.tolist()])),
            "n_bytes": int(t.n_bytes),
            "offset_from_data_start": int(t.data_offset - r.tensors[0].data_offset
                                          + r.tensors[0].field.parts[-1][0] * 0),
            "absolute_data_offset": int(t.data_offset),
        })

    with open(path, "rb") as f:
        head = f.read(4 + 4 + 8 + 8)
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", head)

    truth = {
        "file": "slice.gguf",
        "file_bytes": os.path.getsize(path),
        "magic": magic.decode(),
        "version": int(version),
        "tensor_count": int(n_tensors),
        "kv_count": int(n_kv),
        "alignment": kv.get("general.alignment", {}).get("value", 32),
        "kv": kv,
        "tensors": tensors,
        "recorded_by": "gguf python package",
    }
    with open(os.path.join(FIX, "container_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)

    # One damaged copy: the last tensor's byte offset is pushed past the end of
    # the file. Nothing else changes, so a validator that only checks the magic
    # and the version accepts it and the server dies on the mmap.
    with open(path, "rb") as f:
        blob = bytearray(f.read())
    last = r.tensors[-1]
    off_field = last.field.parts[-1]
    marker = struct.pack("<Q", int(last.field.parts[-1][0]))
    # The tensor-info offset field is the last 8-byte little-endian value written
    # for that tensor before the data section; find it by value, from the end.
    want = struct.pack("<Q", int(off_field[0]))
    idx = blob.rfind(want, 0, r.tensors[0].data_offset)
    damaged_field = None
    if idx >= 0:
        blob[idx:idx + 8] = struct.pack("<Q", os.path.getsize(path) * 4)
        damaged_field = "tensor_info.offset"
    corrupt = os.path.join(FIX, "slice_corrupt.gguf")
    with open(corrupt, "wb") as f:
        f.write(bytes(blob))

    with open(os.path.join(FIX, "corruption_truth.json"), "w") as f:
        json.dump({
            "file": "slice_corrupt.gguf",
            "damaged_field": damaged_field,
            "damaged_tensor": last.name,
            "byte_position": int(idx),
            "original_value": int(off_field[0]),
            "written_value": os.path.getsize(path) * 4,
            "why_it_is_invalid": "tensor data offset points beyond end of file",
        }, f, indent=2)

    print(f"container_truth.json  {len(kv)} kv, {len(tensors)} tensors")
    print(f"slice_corrupt.gguf    damaged {damaged_field} of {last.name} "
          f"at byte {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
