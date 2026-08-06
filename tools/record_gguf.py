#!/usr/bin/env python3
"""Cut a small but genuine GGUF out of a real quantised model.

Several Part-2 units are about the file format itself — reading the tensor
index, dequantising a K-quant block, splicing a chat template into an existing
model. Those units are worthless against a hand-made toy: the interesting parts
of GGUF are exactly the parts a toy gets wrong (alignment padding, the
superblock layout of Q4_K, tokenizer arrays that dwarf the weights).

So the fixture is not synthesised. The key/value section is copied field by
field from a real llama-architecture model, and the tensors are copied
byte-for-byte with their original ggml type. What changes is only how many
tensors come along. The result is a valid GGUF a few megabytes wide that
llama.cpp's own reader accepts.

Ground truth for the dequantisation units comes from gguf.quants.dequantize,
recorded next to the file so the learner's code is checked against the
reference implementation rather than against my reading of the spec.

    python3 tools/record_gguf.py --source <blob> --out projects/_fixtures/gguf
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
except ImportError:
    sys.exit("needs the `gguf` package: pip install gguf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One tensor of each quantisation the model actually uses, plus the F32 norms
# that sit beside them, plus a full first block so the layer-walking units have
# something with real structure to walk.
WANT_PREFIX = ("blk.0.", "token_embd", "output_norm")


def kv_pairs(reader):
    """Every field of the source, as (key, GGUFValueType, python value)."""
    out = []
    for key, field in reader.fields.items():
        if not field.types:
            continue
        head = field.types[0]
        try:
            val = field.contents()
        except Exception:
            continue
        if val is None:
            continue
        if head == GGUFValueType.ARRAY:
            sub = field.types[1] if len(field.types) > 1 else GGUFValueType.STRING
            out.append((key, head, sub, val))
        else:
            out.append((key, head, None, val))
    return out


def pick(reader, budget_bytes):
    """Tensors to carry over: one per ggml type, then the first block, in size order."""
    by_type = {}
    for t in reader.tensors:
        by_type.setdefault(int(t.tensor_type), []).append(t)
    chosen, seen = [], set()
    for _, group in sorted(by_type.items()):
        group.sort(key=lambda t: t.n_bytes)
        for t in group:
            if t.n_bytes and t.name not in seen:
                chosen.append(t)
                seen.add(t.name)
                break
    total = sum(t.n_bytes for t in chosen)
    rest = sorted((t for t in reader.tensors if t.name not in seen),
                  key=lambda t: t.n_bytes)
    for t in rest:
        if not t.name.startswith(WANT_PREFIX):
            continue
        if total + t.n_bytes > budget_bytes:
            continue
        chosen.append(t)
        seen.add(t.name)
        total += t.n_bytes
    return chosen, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "projects", "_fixtures", "gguf"))
    ap.add_argument("--budget-mb", type=float, default=6.0)
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--max-array", type=int, default=2048)
    a = ap.parse_args()

    r = GGUFReader(a.source)
    arch_field = r.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field else "llama"
    chosen, total = pick(r, int(a.budget_mb * 1024 * 1024))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "slice.gguf")

    w = GGUFWriter(path, arch)
    skipped, truncated = 0, {}
    for key, head, sub, val in kv_pairs(r):
        if key == "general.architecture":
            continue
        try:
            if head == GGUFValueType.ARRAY:
                items = list(val)
                if a.max_array and len(items) > a.max_array:
                    truncated[key] = [len(items), a.max_array]
                    items = items[:a.max_array]
                w.add_array(key, items)
            else:
                w.add_key_value(key, val, head)
        except Exception:
            skipped += 1

    w.add_string("general.fixture_note",
                 "cut from a real model: tensor subset, rows truncated, long "
                 "arrays truncated. Format, types and weight bytes are original. "
                 "See manifest.json for exactly what was cut.")

    payload = []
    for t in chosen:
        qt = gguf.GGMLQuantizationType(int(t.tensor_type))
        # The reader reports the logical shape in ggml order; add_tensor_info wants
        # the byte shape in numpy order and undoes the quantisation itself.
        elems = [int(x) for x in reversed(t.shape.tolist())]
        data = t.data
        # K-quants block along the row, so dropping whole rows leaves every
        # surviving superblock byte-identical to the model it came from.
        if a.max_rows and len(elems) > 1 and elems[0] > a.max_rows:
            keep = a.max_rows
            data = data.reshape(elems[0], -1)[:keep]
            truncated[t.name] = [elems[0], keep]
            elems = [keep] + elems[1:]
            data = data.reshape(-1)
        shape = gguf.quants.quant_shape_to_byte_shape(elems, qt) if qt.name not in (
            "F32", "F16", "BF16") else elems
        w.add_tensor_info(t.name, list(shape), data.dtype, int(data.nbytes),
                          raw_dtype=qt)
        payload.append(data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for data in payload:
        w.write_tensor_data(data)
    w.close()

    # Read the result back with the same library the learner will use, and record
    # dequantised ground truth for the quantised tensors we carried over.
    back = GGUFReader(path)
    truth, manifest = {}, []
    for t in back.tensors:
        tt = gguf.GGMLQuantizationType(int(t.tensor_type))
        row = {"name": t.name, "ggml_type": tt.name, "ggml_type_id": int(t.tensor_type),
               "shape": [int(x) for x in t.shape.tolist()], "n_bytes": int(t.n_bytes)}
        try:
            d = gguf.quants.dequantize(t.data, tt).astype(np.float32)
            truth[t.name] = d
            row["dequant_sha_prefix"] = float(np.abs(d).sum())
        except Exception as e:
            row["dequant_error"] = str(e)[:80]
        manifest.append(row)

    np.savez_compressed(os.path.join(a.out, "dequantized_truth.npz"), **truth)
    meta = {
        "source_arch": arch,
        "source_tensors": len(r.tensors),
        "kept_tensors": len(chosen),
        "kv_fields_copied": len(kv_pairs(r)) - 1,
        "kv_fields_skipped": skipped,
        "tensor_bytes": total,
        "truncated": truncated,
        "file_bytes": os.path.getsize(path),
        "tensors": manifest,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{path}  {os.path.getsize(path)/1e6:.1f} MB")
    print(f"  {len(chosen)} tensors of {len(r.tensors)}, "
          f"{meta['kv_fields_copied']} kv fields ({skipped} skipped)")
    print(f"  reread ok: {len(back.tensors)} tensors, "
          f"{len(truth)} dequantised into ground truth")
    for row in manifest[:8]:
        print(f"   {row['name']:32} {row['ggml_type']:8} {row['shape']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
