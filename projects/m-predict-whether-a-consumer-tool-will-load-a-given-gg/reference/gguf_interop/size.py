BLOCK_SIZE_MAP = {
    "F32": 1,
    "F16": 1,
    "Q8_0": 32,
    "Q4_K_M": 256,
    "Q4_0": 32,
    "Q2_K": 256,
}

BYTES_PER_BLOCK = {
    "F32": 4,
    "F16": 2,
    "Q8_0": 34,
    "Q4_K_M": 144,
    "Q4_0": 18,
    "Q2_K": 84,
}


def predict_quant_file_size(tensor_infos: list[dict], quant_type: str, alignment: int = 32) -> int:
    header_and_meta_bytes = 1024
    offset = header_and_meta_bytes

    for t in tensor_infos:
        n_elements = 1
        for d in t["shape"]:
            n_elements *= d

        is_token_emb = t.get("is_embedding", False)
        is_output_norm = t.get("is_norm", False)

        if is_token_emb or is_output_norm:
            t_quant = "F16"
        else:
            t_quant = quant_type

        block_size = BLOCK_SIZE_MAP[t_quant]
        bytes_per_block = BYTES_PER_BLOCK[t_quant]

        n_blocks = (n_elements + block_size - 1) // block_size
        raw_bytes = n_blocks * bytes_per_block

        padding = (alignment - (offset % alignment)) % alignment
        offset += padding + raw_bytes

    return offset
