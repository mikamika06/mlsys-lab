def allocate_slots(num_ctx, parallel):
    return [{"slot_id": i, "ctx_len": num_ctx // parallel} for i in range(parallel)]
