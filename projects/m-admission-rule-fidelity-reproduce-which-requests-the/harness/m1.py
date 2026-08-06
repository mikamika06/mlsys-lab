import ref
from vllmsched.admission import simulate_admission as learner_simulate
from ref import generate_admission_test_data

def ref_simulate_admission(requests, max_num_seqs, max_num_batched_tokens, max_model_len):
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


def check(workdir):
    out = {"schedules_matched": 0.0}
    requests = generate_admission_test_data()
    max_num_seqs = 4
    max_num_batched_tokens = 150
    max_model_len = 512

    want = ref_simulate_admission(requests, max_num_seqs, max_num_batched_tokens, max_model_len)
    try:
        got = learner_simulate(requests, max_num_seqs, max_num_batched_tokens, max_model_len)
        if got == want:
            out["schedules_matched"] = 1.0
        else:
            out["_note"] = f"Expected schedule {want[:2]}, got {got[:2]}"
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {str(e)}"

    return out
