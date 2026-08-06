def compute_utilization(log, max_total_tokens):
    if not log:
        return 0.0

    total_util = 0.0
    for tick in log:
        active_sum = sum(req["input_len"] + req["generated_len"] for req in tick["active"])
        prefill_sum = sum(req["input_len"] for req in tick["prefill"])
        total_util += (active_sum + prefill_sum) / max_total_tokens

    return total_util / len(log)
