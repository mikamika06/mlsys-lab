def simulate_admission(requests, max_num_seqs, max_num_batched_tokens, max_model_len):
    """
    Simulates step-by-step admission of requests into a batch based on vLLM rules.
    """
    schedule = []
    waiting = [dict(r) for r in requests]
    running = []

    while waiting or running:
        admitted_this_step = []
        current_batched_tokens = 0
        current_seqs = 0

        for req in running:
            current_seqs += 1
            current_batched_tokens += 1
            admitted_this_step.append(req["id"])

        next_waiting = []
        for req in waiting:
            req_prompt_len = req["prompt_len"]
            if req_prompt_len > max_model_len:
                continue

            if (current_seqs + 1 <= max_num_seqs and 
                current_batched_tokens + req_prompt_len <= max_num_batched_tokens):
                current_seqs += 1
                current_batched_tokens += req_prompt_len
                admitted_this_step.append(req["id"])
                running.append(req)
            else:
                next_waiting.append(req)

        waiting = next_waiting

        next_running = []
        for req in running:
            req["remaining_output"] -= 1
            if req["remaining_output"] > 0:
                next_running.append(req)
        running = next_running

        if admitted_this_step:
            schedule.append(admitted_this_step)

    return schedule
