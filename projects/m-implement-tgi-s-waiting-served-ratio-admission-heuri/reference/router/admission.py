def admit(queue, active, max_total_tokens, max_prefill_tokens, waiting_served_ratio):
    if not queue:
        return []

    if active and len(queue) <= waiting_served_ratio * len(active):
        return []

    admitted_ids = []
    prefill_sum = 0
    active_sum = sum(req["input_len"] + req["generated_len"] for req in active)

    for req in queue:
        if prefill_sum + req["input_len"] > max_prefill_tokens:
            break
        if active_sum + prefill_sum + req["input_len"] > max_total_tokens:
            break

        admitted_ids.append(req["id"])
        prefill_sum += req["input_len"]

    return admitted_ids
