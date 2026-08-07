def measure_adamw8bit_delta(fp32_state_dict, block_size=2048):
    fp32_bytes = 0
    quant_bytes = 0

    state = fp32_state_dict.get("state", {})
    for p_id, p_state in state.items():
        if not isinstance(p_state, dict):
            continue
        for k, v in p_state.items():
            if isinstance(v, dict) and "shape" in v and "dtype" in v:
                elems = 1
                for d in v["shape"]:
                    elems *= d
                fp32_bytes += elems * 4

                q_state = elems * 1
                num_blocks = (elems + block_size - 1) // block_size
                q_state += num_blocks * 4
                quant_bytes += q_state

    return {
        "fp32_bytes": fp32_bytes,
        "quant_bytes": quant_bytes,
        "delta_bytes": fp32_bytes - quant_bytes,
    }
