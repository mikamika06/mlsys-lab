from vllmsched.admission import simulate_admission

def test_admission_capacity_limits():
    requests = [
        {"id": 1, "prompt_len": 50, "remaining_output": 2},
        {"id": 2, "prompt_len": 60, "remaining_output": 2},
        {"id": 3, "prompt_len": 20, "remaining_output": 2},
    ]
    max_num_seqs = 2
    max_num_batched_tokens = 80
    max_model_len = 512

    schedule = simulate_admission(requests, max_num_seqs, max_num_batched_tokens, max_model_len)
    
    assert len(schedule) > 0
    step1_admitted = schedule[0]
    assert len(step1_admitted) <= max_num_seqs
    assert 1 in step1_admitted
    assert 2 not in step1_admitted
    assert 3 in step1_admitted
