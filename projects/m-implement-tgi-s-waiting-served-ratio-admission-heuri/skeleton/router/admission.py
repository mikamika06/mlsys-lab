def admit(queue, active, max_total_tokens, max_prefill_tokens, waiting_served_ratio):
    """
    queue: list of dicts {"id": str, "input_len": int}
    active: list of dicts {"id": str, "input_len": int, "generated_len": int}
    """
    raise NotImplementedError
