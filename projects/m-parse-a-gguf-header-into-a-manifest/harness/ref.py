import struct

GGUF_MAGIC = b"GGUF"


def encode_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def encode_value(val, vtype) -> bytes:
    if vtype == 0:
        return struct.pack("<B", val)
    elif vtype == 1:
        return struct.pack("<b", val)
    elif vtype == 2:
        return struct.pack("<H", val)
    elif vtype == 3:
        return struct.pack("<h", val)
    elif vtype == 4:
        return struct.pack("<I", val)
    elif vtype == 5:
        return struct.pack("<i", val)
    elif vtype == 6:
        return struct.pack("<f", val)
    elif vtype == 7:
        return struct.pack("<B", 1 if val else 0)
    elif vtype == 8:
        return encode_string(val)
    elif vtype == 10:
        return struct.pack("<Q", val)
    elif vtype == 11:
        return struct.pack("<q", val)
    elif vtype == 12:
        return struct.pack("<d", val)
    elif vtype == 9:
        elem_type, arr = val
        out = struct.pack("<IQ", elem_type, len(arr))
        for item in arr:
            out += encode_value(item, elem_type)
        return out
    else:
        raise ValueError("Unsupported type")


def build_gguf_fixture(alignment=32, metadata=None, tensors=None) -> bytes:
    if metadata is None:
        metadata = {}
    if tensors is None:
        tensors = []

    full_metadata = dict(metadata)
    full_metadata["general.alignment"] = (4, alignment)

    hdr = struct.pack("<4sIII", GGUF_MAGIC, 3, len(tensors), len(full_metadata))

    for k, (vtype, vval) in full_metadata.items():
        hdr += encode_string(k)
        hdr += struct.pack("<I", vtype)
        hdr += encode_value(vval, vtype)

    tensor_info = b""
    current_relative_offset = 0

    for name, dims, ttype, data_bytes in tensors:
        hdr += encode_string(name)
        hdr += struct.pack("<I", len(dims))
        hdr += struct.pack(f"<{len(dims)}Q", *dims)
        hdr += struct.pack("<I", ttype)

        hdr += struct.pack("<Q", current_relative_offset)
        current_relative_offset += len(data_bytes)

    rem = len(hdr) % alignment
    data_offset = len(hdr) if rem == 0 else len(hdr) + (alignment - rem)

    data_section = b""
    curr_pos = data_offset
    for name, dims, ttype, data_bytes in tensors:
        align_rem = curr_pos % alignment
        if align_rem != 0:
            pad = alignment - align_rem
            data_section += b"\x00" * pad
            curr_pos += pad
        data_section += data_bytes
        curr_pos += len(data_bytes)

    header_padding = data_offset - len(hdr)
    return hdr + (b"\x00" * header_padding) + data_section


def parse_gguf_header(data: bytes) -> dict:
    from gguf_parser.header import parse_gguf_header as reference_parse

    return reference_parse(data)


def compute_container_overhead(data: bytes) -> dict:
    from gguf_parser.overhead import (
        compute_container_overhead as reference_compute,
    )

    return reference_compute(data)


def generate_test_fixtures():
    fixtures = []

    f1 = build_gguf_fixture(
        alignment=32,
        metadata={
            "general.architecture": (8, "llama"),
            "llama.block_count": (4, 12),
            "llama.embedding_length": (4, 4096),
        },
        tensors=[
            ("token_embd.weight", [4096, 32000], 0, b"\x00" * (4096 * 32000 * 4)),
            ("blk.0.attn_q.weight", [4096, 4096], 0, b"\x00" * (4096 * 4096 * 4)),
        ],
    )
    fixtures.append({"binary": f1, "alignment": 32})

    f2 = build_gguf_fixture(
        alignment=64,
        metadata={
            "general.name": (8, "test-model"),
            "general.file_type": (4, 1),
            "tokenizer.ggml.tokens": (9, (8, ["<pad>", "<s>", "</s>"])),
            "tokenizer.ggml.scores": (9, (6, [0.0, 1.0, 2.0])),
        },
        tensors=[
            ("output.weight", [32000, 4096], 0, b"\x00" * (32000 * 4096 * 4)),
        ],
    )
    fixtures.append({"binary": f2, "alignment": 64})

    f3 = build_gguf_fixture(
        alignment=32,
        metadata={
            "nested.dims": (9, (9, [(4, [1, 2]), (4, [3, 4])])),
            "flags": (9, (7, [True, False, True])),
        },
        tensors=[],
    )
    fixtures.append({"binary": f3, "alignment": 32})

    f4 = build_gguf_fixture(
        alignment=128,
        metadata={
            "general.author": (8, "MLSys"),
            "general.version": (10, 1000),
        },
        tensors=[
            ("tensor.a", [100], 0, b"\x00" * 400),
            ("tensor.b", [50], 0, b"\x00" * 200),
        ],
    )
    fixtures.append({"binary": f4, "alignment": 128})

    f5 = build_gguf_fixture(
        alignment=32,
        metadata={
            "some.ratio": (12, 3.1415926535),
            "some.int64": (11, -900000000000),
        },
        tensors=[
            ("small.tensor", [10, 10], 0, b"\x00" * 400),
        ],
    )
    fixtures.append({"binary": f5, "alignment": 32})

    return fixtures


GENERATED_FIXTURES = generate_test_fixtures()
