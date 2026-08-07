def configure_slots(max_ctx, slot_count, kv_type):
    if slot_count <= 0 or max_ctx <= 0:
        raise ValueError("Invalid parameters")
    slot_size = max_ctx // slot_count
    bits = 4 if kv_type == "q4_0" else (8 if kv_type == "q8_0" else 16)
    return {"slot_size": slot_size, "kv_bits": bits, "status": "configured"}
