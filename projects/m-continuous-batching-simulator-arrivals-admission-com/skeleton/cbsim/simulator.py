class Request:
    def __init__(self, req_id, arrival_time, prompt_len, gen_len):
        raise NotImplementedError

def admission_predicate(active_requests, candidate_request, max_batch_size, max_capacity):
    raise NotImplementedError

def simulate_continuous(requests, max_batch_size, max_capacity):
    raise NotImplementedError

def simulate_static(requests, max_batch_size, max_capacity):
    raise NotImplementedError

def compute_throughput_ratio(requests, max_batch_size, max_capacity):
    raise NotImplementedError
