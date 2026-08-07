from gguf_parser.header import parse_gguf_header

TYPE_BLOCK_SIZES = {
    0: (1, 4),
    1: (1, 2),
    2: (1, 2),
    3: (1, 2),
    4: (1, 2),
    5: (1, 2),
    6: (1, 2),
    7: (1, 2),
    8: (1, 2),
    9: (1, 2),
    10: (1, 2),
    11: (1, 2),
    12: (1, 2),
}


def _get_tensor_data_size(tensor: dict) -> int:
    num_elements = 1
    for d in tensor["dimensions"]:
        num_elements *= d
    ttype = tensor["type"]
    if ttype in TYPE_BLOCK_SIZES:
        block_len, block_bytes = TYPE_BLOCK_SIZES[ttype]
        blocks = (num_elements + block_len - 1) // block_len
        return blocks * block_bytes
    return num_elements * 4


def compute_container_overhead(data: bytes) -> dict:
    parsed = parse_gguf_header(data)
    alignment = parsed["metadata"].get("general.alignment", 32)
    header_size = parsed["header_size"]

    rem = header_size % alignment
    data_offset = header_size if rem == 0 else header_size + (alignment - rem)
    header_padding = data_offset - header_size

    total_tensor_raw = 0
    total_alignment_waste = 0
    curr_pos = data_offset

    for t in parsed["tensors"]:
        expected_pos = data_offset + t["offset"]
        align_pad = expected_pos - curr_pos
        total_alignment_waste += align_pad
        curr_pos = expected_pos

        t_size = _get_tensor_data_size(t)
        total_tensor_raw += t_size
        curr_pos += t_size

    total_overhead = data_offset + total_alignment_waste

    return {
        "header_size": header_size,
        "data_offset": data_offset,
        "header_padding": header_padding,
        "alignment_waste": total_alignment_waste,
        "total_overhead": total_overhead,
        "raw_tensor_bytes": total_tensor_raw,
        "total_file_bytes": len(data),
    }
